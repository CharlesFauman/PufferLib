# NOTE: This is a Mojo reimplementation of the core game logic from_ the C files.
# It intentionally omits Raylib rendering; render() is a no-op. The focus is on
# parity with your step/reset/game-rules behavior so it can be called from_ Python
# via `mojo.importer`.

# --- Constants (mirroring the C defines) ---
alias EMPTY: Int32 = 0
alias AGENT: Int32 = 1
alias OPPONENT: Int32 = 3
alias AGENT_PAWN: Int32 = 1
alias AGENT_KING: Int32 = 2
alias OPPONENT_PAWN: Int32 = 3
alias OPPONENT_KING: Int32 = 4

struct Log(Copyable, Movable):
    var perf: Float32
    var score: Float32
    var episode_return: Float32
    var episode_length: Float32
    var winrate: Float32
    var n: Float32

    fn __init__(out self):
        self.perf = 0.0
        self.score = 0.0
        self.episode_return = 0.0
        self.episode_length = 0.0
        self.winrate = 0.0
        self.n = 0.0

@fieldwise_init
struct Position(Copyable, Movable):
    var r: Int32
    var c: Int32

@fieldwise_init
struct Move(Copyable, Movable):
    var from_: Position
    var to: Position

@always_inline
fn clamp(val: Float32, low: Float32, high: Float32) -> Float32:
    return Float32(min(max(val, low), high))

# The environment state.
struct Checkers(Copyable, Movable):
    # Buffers
    var observations: List[UInt8]
    var actions: List[Int32]          # length >= 1 (single-discrete action)
    var rewards: List[Float32]        # length >= 1
    var terminals: List[UInt8]        # length >= 1 (0/1)

    # Config/state
    var size: Int32
    var tick: Int32
    var current_player: Int32

    # Cached counts/flags
    var agent_pieces: Int32
    var opponent_pieces: Int32

    # Logging
    var log: Log

    fn __init__(out self,
                 observations: List[UInt8],
                 actions: List[Int32],
                 rewards: List[Float32],
                 terminals: List[UInt8],
                 size: Int32):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.size = size
        self.tick = 0

        self.current_player = AGENT

        self.agent_pieces = 0
        self.opponent_pieces = 0

        self.log = Log()

    # --- Helpers ---
    @always_inline
    fn p2i(self, p: Position) -> Int32:
        return p.r * self.size + p.c

    @always_inline
    fn in_bounds(self, p: Position) -> Bool:
        return p.r >= 0 and p.r < self.size and p.c >= 0 and p.c < self.size

    @always_inline
    fn get_piece(self, p: Position) -> Int32:
        if not self.in_bounds(p):
            return EMPTY
        idx = self.p2i(p)
        return Int32(self.observations[idx])

    @always_inline
    fn get_piece_type(self, p: Position) -> Int32:
        piece = self.get_piece(p)
        if piece == AGENT_PAWN or piece == AGENT_KING:
            return AGENT
        if piece == OPPONENT_PAWN or piece == OPPONENT_KING:
            return OPPONENT
        return EMPTY

    @always_inline
    fn move_direction(self, m: Move) -> Int32:
        return 1 if m.to.r > m.from_.r else -1

    @always_inline
    fn valid_move_direction(self, m: Move) -> Bool:
        piece = self.get_piece(m.from_)
        if piece == AGENT_PAWN:
            return self.move_direction(m) == 1
        if piece == OPPONENT_PAWN:
            return self.move_direction(m) == -1
        return True

    @always_inline
    fn is_diagonal_move(self, m: Move) -> Bool:
        dr = m.to.r - m.from_.r
        dc = m.to.c - m.from_.c
        return (dr == dc) or (dr == -dc)

    @always_inline
    fn move_size(self, m: Move) -> Int32:
        return abs(m.from_.r - m.to.r)

    fn decode_action(self, action: Int32) -> Move:
        num_move_types: Int32 = 8
        pos: Int32 = action / num_move_types
        move_type: Int32 = action % num_move_types

        var m = Move(from_=Position(r=pos / self.size, c=pos % self.size), to=Position(r=0, c=0))

        if move_type == 0:
            m.to.r = m.from_.r - 1; m.to.c = m.from_.c - 1
        elif move_type == 1:
            m.to.r = m.from_.r - 1; m.to.c = m.from_.c + 1
        elif move_type == 2:
            m.to.r = m.from_.r + 1; m.to.c = m.from_.c - 1
        elif move_type == 3:
            m.to.r = m.from_.r + 1; m.to.c = m.from_.c + 1
        elif move_type == 4:
            m.to.r = m.from_.r - 2; m.to.c = m.from_.c - 2
        elif move_type == 5:
            m.to.r = m.from_.r - 2; m.to.c = m.from_.c + 2
        elif move_type == 6:
            m.to.r = m.from_.r + 2; m.to.c = m.from_.c - 2
        else:
            m.to.r = m.from_.r + 2; m.to.c = m.from_.c + 2
        return m

    fn is_valid_move_no_capture(self, m: Move) -> Bool:
        if not self.in_bounds(m.from_) or not self.in_bounds(m.to):
            return False
        if self.get_piece_type(m.from_) != self.current_player:
            return False
        if self.get_piece(m.to) != EMPTY:
            return False
        if not self.valid_move_direction(m):
            return False
        if not self.is_diagonal_move(m):
            return False
        ms = self.move_size(m)
        if ms != 1 and ms != 2:
            return False
        if ms == 2:
            other = AGENT if self.current_player == OPPONENT else OPPONENT
            between = Position(r=(m.from_.r + m.to.r) // 2, c=(m.from_.c + m.to.c) // 2)
            if self.get_piece_type(between) != other:
                return False
        return True

    fn capture_available(self) -> Bool:
        current_pawn = AGENT_PAWN if self.current_player == AGENT else OPPONENT_PAWN
        current_king = AGENT_KING if self.current_player == AGENT else OPPONENT_KING

        for i in range(self.size * self.size):
            piece = Int32(self.observations[i])
            if piece != current_pawn and piece != current_king:
                continue
            r = i / self.size
            c = i % self.size

            dirs = [( -2, -2), (-2,  2), ( 2, -2), ( 2,  2)]
            for d in dirs:
                new_r = r + d[0]
                new_c = c + d[1]
                if new_r < 0 or new_r >= self.size or new_c < 0 or new_c >= self.size:
                    continue
                if self.observations[new_r * self.size + new_c] != 0:
                    continue
                mid_r = r + d[0] // 2
                mid_c = c + d[1] // 2
                mid_piece = Int32(self.observations[mid_r * self.size + mid_c])
                opp_pawn = OPPONENT_PAWN if self.current_player == AGENT else AGENT_PAWN
                opp_king = OPPONENT_KING if self.current_player == AGENT else AGENT_KING
                if mid_piece == opp_pawn or mid_piece == opp_king:
                    move_dir = 1 if d[0] > 0 else -1
                    valid_dir = 1 if self.current_player == AGENT else -1
                    if move_dir != valid_dir:
                        continue
                    return True
        return False

    fn is_valid_move(self, m: Move) -> Bool:
        if self.capture_available() and self.move_size(m) != 2:
            return False
        return self.is_valid_move_no_capture(m)

    fn update_piece_counts(mut self):
        var a: Int32 = 0
        var o: Int32 = 0
        for i in range(self.size * self.size):
            piece = Int32(self.observations[i])
            if piece == AGENT_PAWN or piece == AGENT_KING:
                a += 1
            elif piece == OPPONENT_PAWN or piece == OPPONENT_KING:
                o += 1
        self.agent_pieces = a
        self.opponent_pieces = o

    fn try_make_king(mut self) -> Bool:
        var promoted = False
        # Top row for opponent pawns -> kings
        for i in range(self.size):
            if self.observations[i] == UInt8(OPPONENT_PAWN):
                self.observations[i] = UInt8(OPPONENT_KING)
                promoted = True
        # Bottom row for agent pawns -> kings
        for i in range(self.size):
            idx = self.size * (self.size - 1) + i
            if self.observations[idx] == UInt8(AGENT_PAWN):
                self.observations[idx] = UInt8(AGENT_KING)
                promoted = True
        return promoted

    fn num_pieces_by_player(self, player: Int32) -> Int32:
        return self.agent_pieces if player == AGENT else self.opponent_pieces

    fn is_game_over(self) -> Bool:
        cur_p = self.num_pieces_by_player(self.current_player)
        other = AGENT if self.current_player == OPPONENT else OPPONENT
        oth_p = self.num_pieces_by_player(other)
        if cur_p == 0 or oth_p == 0:
            return True
        if self.capture_available():
            return False
        # check any simple move available
        current_pawn = AGENT_PAWN if self.current_player == AGENT else OPPONENT_PAWN
        current_king = AGENT_KING if self.current_player == AGENT else OPPONENT_KING
        for i in range(self.size * self.size):
            piece = Int32(self.observations[i])
            if piece != current_pawn and piece != current_king:
                continue
            r = i / self.size
            c = i % self.size
            dirs = [(-1,-1),(-1,1),(1,-1),(1,1)]
            for d in dirs:
                nr = r + d[0]
                nc = c + d[1]
                if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
                    continue
                if self.observations[nr*self.size+nc] != 0:
                    continue
                if piece == current_pawn:
                    move_dir = 1 if d[0] > 0 else -1
                    valid_dir = 1 if self.current_player == AGENT else -1
                    if move_dir != valid_dir:
                        continue
                return False
        return True

    fn get_winner(self) -> Int32:
        if self.agent_pieces == 0:
            return OPPONENT
        if self.opponent_pieces == 0:
            return AGENT
        if self.is_game_over():
            return OPPONENT if self.current_player == AGENT else AGENT
        return EMPTY

    fn evaluate_position(self) -> Float32:
        var score: Float32 = 0.0
        for i in range(self.size * self.size):
            piece = Int32(self.observations[i])
            r = i / self.size
            if piece == AGENT_PAWN:
                score += 1.0 + Float32(r) * 0.1
            elif piece == AGENT_KING:
                score += 2.0
            elif piece == OPPONENT_PAWN:
                score -= 1.0 + Float32(self.size - 1 - r) * 0.1
            elif piece == OPPONENT_KING:
                score -= 2.0
        return score

    fn make_move(mut self, action: Int32):
        m = self.decode_action(action)
        if not self.is_valid_move(m):
            self.rewards[0] = -1.0
            return
        moving_piece = self.get_piece(m.from_)
        self.observations[self.p2i(m.from_)] = UInt8(EMPTY)
        self.observations[self.p2i(m.to)] = UInt8(moving_piece)

        var capture_occurred = False
        var reward: Float32 = 0.0

        if self.move_size(m) == 2:
            between = Position(r=(m.from_.r + m.to.r) // 2, c=(m.from_.c + m.to.c) // 2)
            captured_piece = Int32(self.observations[self.p2i(between)])
            self.observations[self.p2i(between)] = UInt8(EMPTY)
            capture_occurred = True
            if captured_piece == AGENT_PAWN or captured_piece == AGENT_KING:
                self.agent_pieces -= 1
                reward -= 0.05
            elif captured_piece == OPPONENT_PAWN or captured_piece == OPPONENT_KING:
                self.opponent_pieces -= 1

        promoted = self.try_make_king()
        if capture_occurred and self.current_player == OPPONENT:
            reward += 0.1
        elif self.current_player == AGENT:
            reward += 0.01

        if self.move_size(m) == 1 or not self.capture_available():
            self.current_player = OPPONENT if self.current_player == AGENT else AGENT

        if promoted:
            # Bonus if an agent king exists on bottom row
            for i in range(self.size):
                idx = self.size * (self.size - 1) + i
                if self.observations[idx] == UInt8(AGENT_KING):
                    reward += 0.05
                    break

        if self.is_game_over():
            self.terminals[0] = 1
            winner = self.get_winner()
            reward = 1 if winner == AGENT else -1

        self.rewards[0] = clamp(reward, -1, 1)

    # --- API matching the C names ---
    fn c_reset(mut self):
        self.tick = 0
        self.terminals[0] = 0
        self.rewards[0] = 0.0

        tiles = self.size * self.size
        for i in range(tiles):
            self.observations[i] = 0

        # place agent pawns in rows 0..2 on dark squares
        for i in range(3):
            for j in range(self.size):
                if ((i + j) % 2) == 1:
                    self.observations[i * self.size + j] = UInt8(AGENT_PAWN)
        # place opponent pawns in last 3 rows on dark squares
        for i in range(self.size - 3, self.size):
            for j in range(self.size):
                if ((i + j) % 2) == 1:
                    self.observations[i * self.size + j] = UInt8(OPPONENT_PAWN)
        self.current_player = AGENT
        self.update_piece_counts()

    fn add_log(mut self):
        self.log.perf += 1 if self.rewards[0] > 0 else 0
        self.log.score += self.evaluate_position()
        self.log.episode_length += Float32(self.tick)
        self.log.episode_return += self.rewards[0]
        if self.terminals[0] == 1:
            self.log.winrate += 1 if self.get_winner() == AGENT else 0
        self.log.n += 1

    fn scripted_random_move(mut self):
        # Simple deterministic pseudo-random scan (no RNG dependency)
        current_pawn = AGENT_PAWN if self.current_player == AGENT else OPPONENT_PAWN
        current_king = AGENT_KING if self.current_player == AGENT else OPPONENT_KING
        has_caps = self.capture_available()
        dirs = [(-1,-1),(-1,1),(1,-1),(1,1),(-2,-2),(-2,2),(2,-2),(2,2)]
        for i in range(self.size * self.size):
            piece = Int32(self.observations[i])
            if piece != current_pawn and piece != current_king:
                continue
            r = i / self.size
            c = i % self.size
            for d_idx in range(len(dirs)):
                d = dirs[d_idx]
                nr = r + d[0]
                nc = c + d[1]
                if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
                    continue
                if self.observations[nr*self.size+nc] != 0:
                    continue
                step_sz = abs(d[0])
                if has_caps and step_sz != 2:
                    continue
                if piece == current_pawn:
                    move_dir = 1 if d[0] > 0 else -1
                    valid_dir = 1 if self.current_player == AGENT else -1
                    if move_dir != valid_dir:
                        continue
                if step_sz == 2:
                    mid_r = r + d[0] // 2
                    mid_c = c + d[1] // 2
                    mid_piece = Int32(self.observations[mid_r*self.size+mid_c])
                    opp_pawn = OPPONENT_PAWN if self.current_player == AGENT else AGENT_PAWN
                    opp_king = OPPONENT_KING if self.current_player == AGENT else AGENT_KING
                    if mid_piece != opp_pawn and mid_piece != opp_king:
                        continue
                action = i * 8 + d_idx
                self.make_move(action)
                return
        # If no move found, leave as-is.

    fn scripted_step(mut self, difficulty: Int32):
        # 0 and 1 both map to a simple policy in this port
        self.scripted_random_move()

    fn c_step(mut self):
        self.tick += 1
        action = self.actions[0]
        self.rewards[0] = 0.0
        self.terminals[0] = 0

        self.make_move(action)
        self.rewards[0] = clamp(self.rewards[0], -1.0, 1.0)
        if self.terminals[0] == 1:
            self.add_log()
            self.c_reset()
            return

        self.scripted_step(1)
        if self.terminals[0] == 1:
            self.add_log()
            self.c_reset()
            return

    fn c_render(self):
        # No-op in Mojo port (Raylib not wired here)
        pass

    fn c_close(self):
        # No dynamic graphics/window state to close in this Mojo port
        pass
# --- Public constructors / helpers for Python interop ---

fn make_env(py_obj: PythonObject) raises -> Checkers:
    size = Int(py_obj)

    tiles = size * size
    env = Checkers(
        observations = List[UInt8](length=UInt(tiles), fill=0),
        actions      = List[Int32](length=1, fill=0),
        rewards      = List[Float32](length=1, fill=0),
        terminals    = List[UInt8](length=1, fill=0),
        size         = size,
    )
    env.c_reset()
    return env

from python import PythonObject
from python.bindings import PythonModuleBuilder
import math
from os import abort

@export
fn PyInit_checkers_mojo() -> PythonObject:
    try:
        var m = PythonModuleBuilder("checkers_mojo")
        m.def_function[make_env]("make_env", docstring="make a new Checkers environment")
        return m.finalize()
    except e:
        return abort[PythonObject](String("error creating Python Mojo module checkers_mojo:", e))