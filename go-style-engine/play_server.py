#!/usr/bin/env python3
"""Minimal local web server to play Go against a trained checkpoint --
built to let a human visually confirm how trans-go-former (the GAB
architecture, §13) actually plays, not just read loss numbers and
tournament win rates off a page.

Deliberately stdlib-only (http.server), no new dependency, matching
this repo's existing minimal-dependency style -- one file, a handful
of JSON endpoints, and an inline HTML/JS board so there's nothing to
build or install beyond what's already in requirements.txt.

Move selection defaults to deterministic play (--dirichlet-epsilon 0,
--temperature 0) -- §13's follow-up investigation found this is what
actually shows the model's real judgment (it reliably finishes games
via double-pass), where temperature=1.0 (needed for self-play TRAINING
diversity, not for real play) was making the same trained model look
like it couldn't finish games, purely from move-selection noise.

Usage:
    .venv/bin/python play_server.py
    .venv/bin/python play_server.py --checkpoint checkpoints/other.pt \\
        --pos-mode both --gab-gen-size 16 --gab-intermediate-dim 16
    .venv/bin/python play_server.py --rounds 400 --port 8765

Then open http://localhost:8000/ in a browser.
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from engine import BoardSpec, TokenEncoder, TokenTransformerNet, ZeroAgent, StyleKnobs, Player, Move
from engine.gotypes import Point
from engine.goboard import GameState
from engine.scoring import compute_game_result, DEFAULT_KOMI

HTML_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>trans-go-former</title>
<style>
  body { font-family: -apple-system, sans-serif; background: #f2ede3; display: flex;
         flex-direction: column; align-items: center; padding: 24px; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  #status { margin: 8px 0 16px; font-size: 14px; color: #333; min-height: 20px; }
  #board { cursor: pointer; }
  #controls { margin-top: 14px; display: flex; gap: 10px; align-items: center; }
  button { padding: 6px 14px; font-size: 14px; cursor: pointer; }
  select { font-size: 14px; }
  #thinking { color: #a55; font-style: italic; }
</style>
</head>
<body>
<h1>trans-go-former &mdash; GAB, 9x9</h1>
<div id="status">Loading...</div>
<canvas id="board" width="480" height="480"></canvas>
<div id="controls">
  <label>Play as:
    <select id="color">
      <option value="black">Black</option>
      <option value="white">White</option>
    </select>
  </label>
  <button id="newGame">New Game</button>
  <button id="pass">Pass</button>
</div>

<script>
const N = 9;
const MARGIN = 30;
const SIZE = 480;
const STEP = (SIZE - 2 * MARGIN) / (N - 1);
const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');

let state = null;

function coordToPixel(i) { return MARGIN + i * STEP; }
function pixelToCoord(px) { return Math.round((px - MARGIN) / STEP); }

function draw() {
  ctx.clearRect(0, 0, SIZE, SIZE);
  ctx.fillStyle = '#dcb35c';
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 1;
  for (let i = 0; i < N; i++) {
    const p = coordToPixel(i);
    ctx.beginPath(); ctx.moveTo(MARGIN, p); ctx.lineTo(SIZE - MARGIN, p); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p, MARGIN); ctx.lineTo(p, SIZE - MARGIN); ctx.stroke();
  }
  if (!state) return;
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const stone = state.stones[r][c];
      if (!stone) continue;
      ctx.beginPath();
      ctx.arc(coordToPixel(c), coordToPixel(r), STEP * 0.42, 0, 2 * Math.PI);
      ctx.fillStyle = stone === 'B' ? '#1a1a1a' : '#f5f5f5';
      ctx.fill();
      ctx.strokeStyle = '#1a1a1a';
      ctx.stroke();
    }
  }
  if (state.last_move && state.last_move !== 'pass') {
    const [r, c] = state.last_move;
    ctx.beginPath();
    ctx.arc(coordToPixel(c), coordToPixel(r), 4, 0, 2 * Math.PI);
    ctx.fillStyle = 'red';
    ctx.fill();
  }
}

function renderStatus() {
  if (!state) return;
  if (state.is_over) {
    const res = state.result;
    statusEl.textContent = res
      ? `Game over -- ${res.winner} wins by ${res.margin.toFixed(1)} (B:${res.black_score} W:${res.white_score}+${res.komi}komi)`
      : 'Game over.';
    return;
  }
  const turn = state.next_player === state.human_color ? 'Your move' : 'AI thinking...';
  let extra = '';
  if (state.ai_diagnostics) {
    const d = state.ai_diagnostics;
    extra = ` (AI: win_prob=${(d.win_prob_black * 100).toFixed(0)}%, complexity=${d.complexity.toFixed(2)}, ${d.think_time_s.toFixed(1)}s)`;
  }
  statusEl.textContent = `${turn}${extra}`;
}

async function newGame() {
  const color = document.getElementById('color').value;
  statusEl.textContent = 'Starting...';
  const resp = await fetch('/api/new', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({human_color: color})
  });
  state = await resp.json();
  draw(); renderStatus();
}

async function sendMove(body) {
  statusEl.textContent = 'AI thinking...';
  const resp = await fetch('/api/move', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  if (resp.status !== 200) {
    const err = await resp.json();
    statusEl.textContent = 'Illegal move: ' + err.error;
    return;
  }
  state = await resp.json();
  draw(); renderStatus();
}

canvas.addEventListener('click', (e) => {
  if (!state || state.is_over || state.next_player !== state.human_color) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (SIZE / rect.width);
  const y = (e.clientY - rect.top) * (SIZE / rect.height);
  const col = pixelToCoord(x);
  const row = pixelToCoord(y);
  if (row < 0 || row >= N || col < 0 || col >= N) return;
  sendMove({row, col});
});

document.getElementById('newGame').addEventListener('click', newGame);
document.getElementById('pass').addEventListener('click', () => sendMove({pass: true}));

newGame();
</script>
</body>
</html>
"""


class GameServer:
    """Holds the one live game + the AI agent. A local single-player
    dev tool -- global mutable state is fine, no concurrent games."""

    def __init__(self, model, encoder, knobs, board_size):
        self.model = model
        self.encoder = encoder
        self.knobs = knobs
        self.board_size = board_size
        self.game = None
        self.human_color = None
        self.last_move = None
        self.ai_diagnostics = None
        self.lock = threading.Lock()

    def new_game(self, human_color):
        self.game = GameState.new_game(self.board_size)
        self.human_color = Player.black if human_color == 'black' else Player.white
        self.last_move = None
        self.ai_diagnostics = None
        if self.game.next_player != self.human_color:
            self._ai_move()

    def _ai_move(self):
        agent = ZeroAgent(self.model, self.encoder, self.knobs)
        mover = self.game.next_player
        move, diag = agent.select_move(self.game)
        # diag['root_value'] is from the mover's own perspective (see
        # engine/mcts.py); convert to win_prob_black the same way
        # selfplay.py's play_one_game does, for the UI to display.
        root_value = diag.get('root_value', 0.0)
        diag['win_prob_black'] = (root_value + 1) / 2 if mover == Player.black \
            else (1 - root_value) / 2
        self.game = self.game.apply_move(move)
        self.last_move = move
        self.ai_diagnostics = diag

    def apply_human_move(self, move):
        if not self.game.is_valid_move(move):
            raise ValueError(f'illegal move: {move}')
        self.game = self.game.apply_move(move)
        self.last_move = move
        self.ai_diagnostics = None
        if not self.game.is_over() and self.game.next_player != self.human_color:
            self._ai_move()

    def to_json(self):
        board = self.game.board
        stones = []
        for r in range(1, self.board_size + 1):
            row = []
            for c in range(1, self.board_size + 1):
                occupant = board.get(Point(row=r, col=c))
                row.append(None if occupant is None else ('B' if occupant == Player.black else 'W'))
            stones.append(row)

        is_over = self.game.is_over()
        result = None
        if is_over and self.game.last_move is not None and not self.game.last_move.is_resign:
            gr = compute_game_result(self.game)
            result = {
                'winner': 'B' if gr.winner == Player.black else 'W',
                'margin': gr.winning_margin,
                'black_score': gr.b,
                'white_score': gr.w,
                'komi': gr.komi,
            }

        last_move_json = None
        if self.last_move is not None:
            if self.last_move.is_pass:
                last_move_json = 'pass'
            elif self.last_move.is_play:
                last_move_json = [self.last_move.point.row - 1, self.last_move.point.col - 1]

        diag_json = None
        if self.ai_diagnostics is not None:
            diag_json = {
                'win_prob_black': self.ai_diagnostics.get('win_prob_black', 0.5),
                'complexity': self.ai_diagnostics.get('complexity', 0.0),
                'think_time_s': self.ai_diagnostics.get('think_time_s', 0.0),
            }

        return {
            'board_size': self.board_size,
            'stones': stones,
            'next_player': 'black' if self.game.next_player == Player.black else 'white',
            'human_color': 'black' if self.human_color == Player.black else 'white',
            'is_over': is_over,
            'result': result,
            'last_move': last_move_json,
            'ai_diagnostics': diag_json,
        }


def make_handler(server_state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # quiet; this is a local dev tool

        def _send_json(self, obj, status=200):
            body = json.dumps(obj).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                body = HTML_PAGE.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == '/api/state':
                with server_state.lock:
                    self._send_json(server_state.to_json())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            if self.path == '/api/new':
                with server_state.lock:
                    server_state.new_game(body.get('human_color', 'black'))
                    self._send_json(server_state.to_json())
                return

            if self.path == '/api/move':
                with server_state.lock:
                    try:
                        if body.get('pass'):
                            move = Move.pass_turn()
                        else:
                            row, col = body['row'], body['col']
                            move = Move.play(Point(row=row + 1, col=col + 1))
                        server_state.apply_human_move(move)
                    except (ValueError, KeyError) as e:
                        self._send_json({'error': str(e)}, status=400)
                        return
                    self._send_json(server_state.to_json())
                return

            self.send_response(404)
            self.end_headers()

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', default='checkpoints/gen_loop_kifu9x9_gab/gen10.pt',
                         help='defaults to the verified GAB checkpoint from §13')
    parser.add_argument('--board-size', type=int, default=9)
    parser.add_argument('--pos-mode', default='gab_absolute')
    parser.add_argument('--gab-gen-size', type=int, default=16)
    parser.add_argument('--gab-intermediate-dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--history-depth', type=int, default=7)
    parser.add_argument('--rounds', type=int, default=200,
                         help='MCTS rounds per AI move -- higher is stronger but slower')
    parser.add_argument('--dirichlet-epsilon', type=float, default=0.0,
                         help='0.0 (default): deterministic play -- §13\'s follow-up found '
                              'this is what actually shows the model finishing games '
                              'reliably. Set >0 for self-play-style exploration instead.')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args(argv)

    spec = BoardSpec(board_size=args.board_size, history_depth=args.history_depth)
    encoder = TokenEncoder(spec)
    model = TokenTransformerNet(
        spec, d_model=64, nhead=4, num_layers=4, dim_feedforward=128,
        dropout=args.dropout, pos_mode=args.pos_mode,
        gab_gen_size=args.gab_gen_size, gab_intermediate_dim=args.gab_intermediate_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()

    knobs = StyleKnobs(rounds_per_move=args.rounds,
                        dirichlet_epsilon=args.dirichlet_epsilon,
                        temperature=args.temperature)

    server_state = GameServer(model, encoder, knobs, args.board_size)
    server_state.new_game('black')

    httpd = ThreadingHTTPServer(('localhost', args.port), make_handler(server_state))
    print(f'trans-go-former play server: http://localhost:{args.port}/')
    print(f'  checkpoint={args.checkpoint} pos_mode={args.pos_mode} rounds={args.rounds}')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
