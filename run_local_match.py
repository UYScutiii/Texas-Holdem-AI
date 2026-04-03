# run_local_match.py

from core.engine import Table, TournamentManager, Seat, InProcessBot
from core.bot_api import BotAdapter, PlayerView, Action
from bots.poker_mind_bot import SmartBot
from bots.monte_carlo_bot import MonteCarloBot
from bots.ml_bot import MLBot
from bots.rl_bot import RLBot  # Add RLBot import
import matplotlib.pyplot as plt
from collections import defaultdict


def run_tournament_until_winner(seats, bots, small_blind, big_blind):
    """
    Run tournament until only one player has chips remaining.
    Returns chip history and final results.
    """
    table = Table()
    tm = TournamentManager(table)
    
    # Track chip history after each hand
    chip_history = []
    hand_count = 0
    dealer_index = 0
    initial_chips = {s.player_id: s.chips for s in seats}
    
    # Store initial state
    chip_history.append({s.player_id: s.chips for s in seats})
    
    print("=" * 60)
    print("TOURNAMENT MODE: Playing until one winner!")
    print("=" * 60)
    
    while True:
        # Count active players (those with chips > 0)
        active_players = [s for s in seats if s.chips > 0]
        
        if len(active_players) <= 1:
            winner = active_players[0].player_id if active_players else None
            if winner:
                print(f"\nTOURNAMENT OVER! Winner: {winner}")
            break
        
        # Play one hand
        hand_count += 1
        print(f"\n--- Hand #{hand_count} ---")
        
        # Only include active players in the hand
        active_seats = [s for s in seats if s.chips > 0]
        
        # Create bot adapters
        # MLBot needs InProcessBot (converts PlayerView to dict)
        # SmartBot and MonteCarloBot already accept PlayerView, so use a pass-through adapter
        class PlayerViewAdapter(BotAdapter):
            def __init__(self, bot):
                self.bot = bot
            def act(self, view: PlayerView) -> Action:
                return self.bot.act(view)
        
        active_bots = {}
        for s in active_seats:
            bot = bots[s.player_id]
            if isinstance(bot, (MLBot, RLBot)):  # Both need InProcessBot
                active_bots[s.player_id] = InProcessBot(bot)
            else:
                # SmartBot and MonteCarloBot already accept PlayerView
                active_bots[s.player_id] = PlayerViewAdapter(bot)
        
        # Play hand
        res = table.play_hand(
            seats=active_seats,
            small_blind=small_blind,
            big_blind=big_blind,
            dealer_index=dealer_index % len(active_seats),
            bot_for=active_bots,
            on_event=None
        )
        
        # Update dealer position
        dealer_index = (dealer_index + 1) % len(seats)
        
        # Store chip state after this hand
        chip_history.append({s.player_id: s.chips for s in seats})
        
        # Print current stacks
        print(f"Stacks after hand {hand_count}:")
        for s in seats:
            if s.chips > 0:
                print(f"  {s.player_id}: {s.chips:.2f} chips")
        
        # Safety limit (prevent infinite loops)
        if hand_count > 10000:
            print("Safety limit reached (10,000 hands). Stopping.")
            break
    
    return chip_history, hand_count


def plot_tournament_progress(chip_history, player_ids):
    """
    Create a visualization of chip stacks over time.
    """
    if not chip_history:
        print("No data to plot.")
        return
    
    # Prepare data for plotting
    hands = list(range(len(chip_history)))
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot each player's chip stack
    for pid in player_ids:
        chips_over_time = [state.get(pid, 0) for state in chip_history]
        plt.plot(hands, chips_over_time, marker='o', label=pid, linewidth=2, markersize=4)
    
    plt.xlabel('Hand Number', fontsize=12)
    plt.ylabel('Chip Stack', fontsize=12)
    plt.title('Tournament Progress: Chip Stacks Over Time', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('output/tournament_progress.png', dpi=150, bbox_inches='tight')
    print(f"\nChart saved as 'output/tournament_progress.png'")
    
    # Show the plot
    plt.show()


def main():
    # 3-player demo table with 500 chips each
    seats = [Seat(player_id=f"P{i+1}", chips=500) for i in range(3)]
    player_ids = [s.player_id for s in seats]
    
    # Set up bots - you can now use RLBot!
    bots = {
        "P1": RLBot(training_mode=False),  # Use trained RL bot
        "P2": SmartBot(),
        "P3": MonteCarloBot(),
    }
    
    # Run tournament until one winner
    chip_history, total_hands = run_tournament_until_winner(
        seats=seats,
        bots=bots,
        small_blind=1,
        big_blind=2
    )
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total hands played: {total_hands}")
    print("\nFinal stacks:")
    for s in seats:
        status = "WINNER" if s.chips > 0 else "Eliminated"
        print(f"  {s.player_id}: {s.chips:.2f} chips - {status}")
    
    # Create visualization
    print("\nGenerating tournament chart...")
    plot_tournament_progress(chip_history, player_ids)


if __name__ == "__main__":
    main()
