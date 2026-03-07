use postflop_solver::*;
use serde::{Deserialize, Serialize};
use std::io::{self, Read};

// ── Input ────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SolverInput {
    /// Board cards, e.g. ["Ah", "Kd", "7s"] or ["Ah", "Kd", "7s", "Qc"]
    board: Vec<String>,
    /// OOP player range string (PioSOLVER format)
    oop_range: String,
    /// IP player range string
    ip_range: String,
    /// Pot size in BB at the start of the current street
    starting_pot: f64,
    /// Effective stack in BB
    effective_stack: f64,
    /// Hero's hole cards, e.g. ["Ah", "Kd"]
    hero_hand: [String; 2],
    /// "oop" or "ip"
    hero_position: String,

    // Optional bet sizing overrides (defaults to standard sizes)
    #[serde(default)]
    bet_sizes_oop: Option<String>,
    #[serde(default)]
    bet_sizes_ip: Option<String>,
    #[serde(default)]
    raise_sizes_oop: Option<String>,
    #[serde(default)]
    raise_sizes_ip: Option<String>,

    /// Max iterations (default 1000)
    #[serde(default = "default_max_iterations")]
    max_iterations: u32,
    /// Target exploitability as fraction of pot (default 0.005 = 0.5%)
    #[serde(default = "default_target_exploitability")]
    target_exploitability: f64,
}

fn default_max_iterations() -> u32 { 1000 }
fn default_target_exploitability() -> f64 { 0.005 }

// ── Output ───────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct SolverOutput {
    actions: Vec<ActionResult>,
    iterations: u32,
    exploitability: f64,
    ev: f64,
    equity: f64,
}

#[derive(Serialize)]
struct ActionResult {
    action: String,
    frequency: f64,
    ev: f64,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn format_action(action: &Action, pot: i32) -> String {
    match action {
        Action::Fold => "Fold".to_string(),
        Action::Check => "Check".to_string(),
        Action::Call => "Call".to_string(),
        Action::Bet(size) => {
            let pct = (*size as f64 / pot as f64) * 100.0;
            if *size > pot * 3 / 2 {
                format!("AllIn({})", size)
            } else {
                format!("Bet {:.0}% ({})", pct, size)
            }
        }
        Action::Raise(size) => {
            format!("Raise({})", size)
        }
        Action::AllIn(size) => format!("AllIn({})", size),
        Action::Chance(_) => "Chance".to_string(),
        _ => format!("{:?}", action),
    }
}

fn find_hero_index(game: &PostFlopGame, player: usize, hand: &[String; 2]) -> Option<usize> {
    let cards = game.private_cards(player);
    let hand_str = format!("{}{}", hand[0], hand[1]);
    let hand_rev = format!("{}{}", hand[1], hand[0]);

    let card_strings = holes_to_strings(cards).ok()?;
    card_strings
        .iter()
        .position(|s| s == &hand_str || s == &hand_rev)
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn run(input: SolverInput) -> Result<SolverOutput, Box<dyn std::error::Error>> {
    // Parse board
    let board = &input.board;
    if board.len() < 3 || board.len() > 5 {
        return Err(format!("Board must have 3-5 cards, got {}", board.len()).into());
    }

    let flop_str = format!("{}{}{}", board[0], board[1], board[2]);
    let flop = flop_from_str(&flop_str)
        .map_err(|_| format!("Invalid flop: {}", flop_str))?;

    let turn = if board.len() >= 4 {
        card_from_str(&board[3]).map_err(|_| format!("Invalid turn: {}", board[3]))?
    } else {
        NOT_DEALT
    };

    let river = if board.len() >= 5 {
        card_from_str(&board[4]).map_err(|_| format!("Invalid river: {}", board[4]))?
    } else {
        NOT_DEALT
    };

    let initial_state = match board.len() {
        3 => BoardState::Flop,
        4 => BoardState::Turn,
        5 => BoardState::River,
        _ => unreachable!(),
    };

    // Parse ranges
    let oop_range: Range = input.oop_range.parse()
        .map_err(|_| format!("Invalid OOP range: {}", input.oop_range))?;
    let ip_range: Range = input.ip_range.parse()
        .map_err(|_| format!("Invalid IP range: {}", input.ip_range))?;

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop,
        turn,
        river,
    };

    // Bet sizes — defaults: 33%/66%/allin for bets, 2.5x/allin for raises
    let oop_bets = input.bet_sizes_oop.as_deref().unwrap_or("33%, 66%, a");
    let oop_raises = input.raise_sizes_oop.as_deref().unwrap_or("2.5x");
    let ip_bets = input.bet_sizes_ip.as_deref().unwrap_or("33%, 66%, a");
    let ip_raises = input.raise_sizes_ip.as_deref().unwrap_or("2.5x");

    let oop_sizes = BetSizeOptions::try_from((oop_bets, oop_raises))
        .map_err(|e| format!("Invalid OOP bet sizes: {:?}", e))?;
    let ip_sizes = BetSizeOptions::try_from((ip_bets, ip_raises))
        .map_err(|e| format!("Invalid IP bet sizes: {:?}", e))?;

    // Scale pot/stacks to integer chips (multiply by 10 for 0.1 BB precision)
    let scale = 10;
    let pot = (input.starting_pot * scale as f64).round() as i32;
    let stack = (input.effective_stack * scale as f64).round() as i32;

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: pot,
        effective_stack: stack,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [oop_sizes.clone(), ip_sizes.clone()],
        turn_bet_sizes: [oop_sizes.clone(), ip_sizes.clone()],
        river_bet_sizes: [oop_sizes, ip_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
    };

    let action_tree = ActionTree::new(tree_config)
        .map_err(|e| format!("Failed to build action tree: {:?}", e))?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .map_err(|e| format!("Failed to create game: {:?}", e))?;

    // Use compression to save memory
    game.allocate_memory(true);

    // Solve
    let target = pot as f32 * input.target_exploitability as f32;
    let exploitability = solve(&mut game, input.max_iterations, target, false);

    // Find hero's hand index
    let hero_player = if input.hero_position == "oop" { 0 } else { 1 };
    let hero_idx = find_hero_index(&game, hero_player, &input.hero_hand)
        .ok_or_else(|| format!(
            "Hero hand {}{} not found in {} range",
            input.hero_hand[0], input.hero_hand[1], input.hero_position
        ))?;

    // Get strategy at root for hero
    game.cache_normalized_weights();
    let equity_all = game.equity(hero_player);
    let ev_all = game.expected_values(hero_player);

    let hero_equity = equity_all[hero_idx] as f64;
    let hero_ev = ev_all[hero_idx] as f64 / scale as f64; // convert back to BB

    let actions = game.available_actions();
    let strategy = game.strategy();
    let num_hands = game.private_cards(hero_player).len();

    let mut action_results = Vec::new();
    for (i, action) in actions.iter().enumerate() {
        let freq = strategy[i * num_hands + hero_idx] as f64;
        // To get per-action EV we'd need to traverse; use overall EV weighted by freq
        action_results.push(ActionResult {
            action: format_action(action, pot),
            frequency: freq,
            ev: 0.0, // per-action EV requires tree traversal (TODO)
        });
    }

    Ok(SolverOutput {
        actions: action_results,
        iterations: input.max_iterations,
        exploitability: exploitability as f64 / scale as f64,
        ev: hero_ev,
        equity: hero_equity,
    })
}

fn main() {
    // Read JSON from stdin
    let mut input_str = String::new();
    io::stdin().read_to_string(&mut input_str).unwrap_or_else(|e| {
        eprintln!("Error reading stdin: {}", e);
        std::process::exit(1);
    });

    let input: SolverInput = serde_json::from_str(&input_str).unwrap_or_else(|e| {
        eprintln!("Error parsing JSON: {}", e);
        std::process::exit(1);
    });

    match run(input) {
        Ok(output) => {
            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }
        Err(e) => {
            eprintln!("Solver error: {}", e);
            std::process::exit(1);
        }
    }
}
