import os
import sys
import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader

# Add project root to path so imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.poker_mlp import PokerMLP

import torch.nn as nn

RANKS = {"2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
         "9":9, "T":10, "J":11, "Q":12, "K":13, "A":14}
SUITS = {"c":0, "d":1, "h":2, "s":3}

STREET_MAP = {"preflop":0, "flop":1, "turn":2, "river":3}


def encode_card(card):
    """Convert ['A','h'] -> [14, 2]"""
    rank, suit = card
    return [RANKS[rank], SUITS[suit]]




def load_decision_logs(root):
    """
    Recursively loads every .jsonl file inside logs/
    and returns a list of decision-dictionaries.
    """
    decisions = []

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith(".jsonl"):
                continue

            full = os.path.join(dirpath, fname)
            try:
                with open(full, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)

                        # Skip final result rows
                        if "chosen_action" not in obj:
                            continue

                        decisions.append(obj)

            except Exception as e:
                print(f"[WARN] Could not read {full}: {e}")

    return decisions

class PokerDataset(Dataset):
    ACTION_MAP = {
        "fold": 0,
        "check": 1,
        "call": 2,
        # raises get bucketed into 3 bins
        "raise_small": 3,
        "raise_medium": 4,
        "raise_large": 5
    }

    def __init__(self, log_folder, filter_players=None, filter_winners_only=False,
                 starting_chips=500):
        """
        Args:
            log_folder: Directory containing log files
            filter_players: List of player IDs to include (e.g., ["P3"] for MonteCarloBot only)
            filter_winners_only: If True, only include decisions from hands that were won
            starting_chips: Used to normalise pot and stack features to [0, ~2] scale
        """
        self.samples = []
        self.starting_chips = max(1, starting_chips)
        # Load all decisions first to build memory context
        all_decisions = []  # Will store all decisions in order
        for path in glob.glob(f"{log_folder}/**/*.jsonl", recursive=True):
            with open(path) as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        if "chosen_action" in row:
                            # Store with file path and line number for ordering
                            row['_file'] = path
                            row['_line'] = line_num
                            all_decisions.append(row)
                    except:
                        continue

        # Load hand results to track winners
        hand_results = {}  # hand_id -> winner player_id
        if filter_winners_only:
            for path in glob.glob(f"{log_folder}/**/*.jsonl", recursive=True):
                with open(path) as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                            if "result" in row and "hand_id" in row:
                                result = row["result"]
                                if result.get("net", 0) > 0:  # Winner (positive net)
                                    hand_results[row["hand_id"]] = result.get("player")
                        except:
                            continue

        # Process each decision with memory context
        for decision_idx, row in enumerate(all_decisions):
            if "chosen_action" not in row:
                continue

            # Filter by player
            player_id = row.get("player")
            if filter_players and player_id not in filter_players:
                continue

            # Filter by winner
            if filter_winners_only:
                hand_id = row.get("hand_id")
                if hand_id not in hand_results or hand_results[hand_id] != player_id:
                    continue

            chosen = row["chosen_action"]
            act_type = chosen["type"]
            act_amt = chosen["amount"]

            # --------- LABEL ENCODING ---------
            if act_type in ("fold", "check", "call"):
                label = self.ACTION_MAP[act_type]

            elif act_type in ("raise", "bet"):
                legal = next(
                    (la for la in row["legal"]
                     if la["type"] in ("raise", "bet")),
                    None
                )
                if legal is None:
                    label = self.ACTION_MAP["call"]
                else:
                    lo, hi = legal["min"], legal["max"]
                    if act_amt <= lo + (hi - lo) * 0.33:
                        label = self.ACTION_MAP["raise_small"]
                    elif act_amt <= lo + (hi - lo) * 0.66:
                        label = self.ACTION_MAP["raise_medium"]
                    else:
                        label = self.ACTION_MAP["raise_large"]
            else:
                continue

            # --------- FEATURE ENCODING ---------
            hole = row.get("hole", [])
            hole_enc = []

            for c in hole:
                hole_enc += encode_card(c)

            while len(hole_enc) < 4:
                hole_enc.append(0)

            board = row["board"]
            board_enc = []
            for c in board:
                board_enc += encode_card(c)
            while len(board_enc) < 10:
                board_enc.append(0)

            street = STREET_MAP[row["street"]]
            # Normalise monetary features to [0, ~2] scale (same as MLBot inference)
            scale = self.starting_chips
            pot = float(row["pot"]) / scale
            to_call = float(row["to_call"]) / scale

            stacks = row["stacks"]
            me = row["player"]
            hero_stack = float(stacks[me]) / scale
            eff_stack = min(float(v) for v in stacks.values()) / scale
            n_players = len([v for v in stacks.values() if v > 0])

            # Hand strength — normalised the same way as MLBot._estimate_hand_strength()
            if hole and len(hole) >= 2:
                from core.engine import eval_hand, EVAL_HAND_MAX
                score = eval_hand(hole, board)
                hand_strength = score / EVAL_HAND_MAX
            else:
                hand_strength = 0.0

            # Pot odds
            if pot + to_call > 0:
                pot_odds = to_call / (pot + to_call)
            else:
                pot_odds = 0.0

            # Position encoding — matches ml_bot.py position_order
            position_order = {
                "UTG": 0.0, "UTG+1": 0.1, "MP": 0.3, "LJ": 0.4,
                "HJ": 0.6, "CO": 0.8, "BTN": 1.0, "SB": 0.5, "BB": 0.5
            }
            position = row.get("position", "MP")
            position_value = position_order.get(position, 0.5)

            # NEW: Calculate memory features from previous decisions
            opponents = row.get("opponents", [])
            memory_features = self._calculate_memory_features(
                all_decisions, decision_idx, me, opponents, row.get("_file")
            )

            features = [
                street, pot, to_call,
                hero_stack, eff_stack, n_players
            ] + hole_enc + board_enc + [hand_strength, pot_odds, position_value] + memory_features

            self.samples.append(
                (
                    torch.tensor(features, dtype=torch.float32),
                    torch.tensor(label)
                )
            )

    def _calculate_memory_features(self, all_decisions, current_idx, me, opponents, current_file):
        """
        Calculate opponent behavior features from previous decisions in the same session.
        Returns [avg_aggression, avg_tightness, avg_vpip]
        """
        if not opponents:
            return [0.5, 0.5, 0.5]  # Neutral values

        # Get previous decisions from same file (same tournament session)
        previous_decisions = [
            d for d in all_decisions[:current_idx]
            if d.get("_file") == current_file and d.get("player") in opponents
        ]

        if not previous_decisions:
            return [0.5, 0.5, 0.5]  # No history yet

        # Calculate stats from last 10 opponent actions
        recent = previous_decisions[-10:]

        total_actions = len(recent)
        if total_actions == 0:
            return [0.5, 0.5, 0.5]

        aggressive_count = sum(1 for d in recent
                              if d.get("chosen_action", {}).get("type") in ("bet", "raise"))
        fold_count = sum(1 for d in recent
                        if d.get("chosen_action", {}).get("type") == "fold")
        vpip_count = sum(1 for d in recent
                        if d.get("chosen_action", {}).get("type") in ("call", "bet", "raise"))

        avg_aggression = aggressive_count / total_actions
        avg_tightness = fold_count / total_actions
        avg_vpip = vpip_count / total_actions

        return [avg_aggression, avg_tightness, avg_vpip]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train_model(
    model,
    train_loader,
    val_loader,
    lr=1e-3,
    epochs=8,
    device="cpu",
    class_weights=None,
    feature_mean=None,
    feature_std=None,
):
    model = model.to(device)

    # Class-weighted loss
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Apply feature normalization
            if feature_mean is not None:
                x = (x - feature_mean.to(device)) / (feature_std.to(device) + 1e-8)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # ---- Validation ----
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                if feature_mean is not None:
                    x = (x - feature_mean.to(device)) / (feature_std.to(device) + 1e-8)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_val = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = 100 * correct / total if total > 0 else 0.0

        # LR scheduling
        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {avg_train:.4f} | "
              f"Val Loss: {avg_val:.4f} | "
              f"Val Acc:  {accuracy:.2f}% | "
              f"LR: {current_lr:.2e}")

        # Model checkpointing: save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_dir = os.path.join(project_root, "models")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(best_state, os.path.join(save_dir, "ml_model_best.pt"))
            print(f"  -> New best model saved (val_loss={avg_val:.4f})")

    # Restore best weights for return
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def compute_confusion_matrix(model, loader, num_classes, device, feature_mean, feature_std):
    """Compute and print per-class accuracy (confusion matrix)."""
    ACTION_NAMES = ["fold", "check", "call", "raise_small", "raise_med", "raise_large"]
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if feature_mean is not None:
                x = (x - feature_mean.to(device)) / (feature_std.to(device) + 1e-8)
            preds = model(x).argmax(dim=1)
            for t, p in zip(y, preds):
                confusion[t.item(), p.item()] += 1

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX  (rows=true, cols=predicted)")
    print("=" * 60)

    header = f"{'':>14s}" + "".join(f"{n:>10s}" for n in ACTION_NAMES)
    print(header)
    for i, name in enumerate(ACTION_NAMES):
        row_total = confusion[i].sum().item()
        acc = 100 * confusion[i, i].item() / row_total if row_total > 0 else 0.0
        row_str = f"{name:>14s}" + "".join(f"{confusion[i,j].item():>10d}" for j in range(num_classes))
        row_str += f"  | acc {acc:5.1f}%  (n={row_total})"
        print(row_str)

    total_correct = sum(confusion[i, i].item() for i in range(num_classes))
    total_samples = confusion.sum().item()
    print(f"\nOverall accuracy: {100*total_correct/total_samples:.2f}% ({total_correct}/{total_samples})")


if __name__ == "__main__":
    import argparse
    import sys
    from torch.utils.data import DataLoader, random_split

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Folder containing JSONL logs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--filter_players", type=str, nargs="+", default=None,
                        help="Only train on decisions from these players (e.g., --filter_players P3)")
    parser.add_argument("--filter_winners", action="store_true",
                        help="Only train on decisions from winning hands")
    args = parser.parse_args()

    print("Loading logs...")
    dataset = PokerDataset(
        log_folder=args.log_dir,
        filter_players=args.filter_players,
        filter_winners_only=args.filter_winners
    )

    if len(dataset) == 0:
        print(f"Error: No training data found in {args.log_dir}")
        sys.exit(1)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch)

    print(f"Training samples: {train_size}  |  Validation samples: {val_size}")
    if args.filter_players:
        print(f"Filtered to players: {args.filter_players}")
    if args.filter_winners:
        print("Filtered to winning hands only")

    sample_x, _ = dataset[0]
    input_size = sample_x.shape[0]

    model = PokerMLP(input_dim=input_size, hidden=128, num_classes=6)

    print("Starting training...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device
    )

    # ---- SAVE MODEL ----
    save_path = "models/ml_model.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print(f"\nModel saved to: {save_path}")
