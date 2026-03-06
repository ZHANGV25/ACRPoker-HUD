"""Network layer: send game state from Mac (OCR) to PC (solver) over ZeroMQ."""

import json
import time
import zmq


class GameStateSender:
    """Sends game state JSON to the solver PC over ZeroMQ PUB socket."""

    def __init__(self, bind_addr: str = "tcp://*:5555"):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.bind(bind_addr)
        # PUB sockets need a moment to set up subscriptions
        time.sleep(0.1)

    def send(self, game_state) -> None:
        """Send a GameState object as JSON."""
        data = game_state.to_solver_input()
        msg = json.dumps(data)
        self.socket.send_string(msg)

    def close(self) -> None:
        self.socket.close()
        self.ctx.term()


class GameStateReceiver:
    """Receives game state JSON from the Mac over ZeroMQ SUB socket.

    Run this on the PC side.
    """

    def __init__(self, connect_addr: str = "tcp://localhost:5555"):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.connect(connect_addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def receive(self, timeout_ms: int = 5000) -> dict:
        """Receive a game state dict. Returns None on timeout."""
        if self.socket.poll(timeout_ms):
            msg = self.socket.recv_string()
            return json.loads(msg)
        return None

    def close(self) -> None:
        self.socket.close()
        self.ctx.term()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or sys.argv[1] not in ("send", "recv"):
        print("Usage:")
        print("  python -m src.network send [addr]  # Mac side (publisher)")
        print("  python -m src.network recv [addr]  # PC side (subscriber)")
        sys.exit(1)

    mode = sys.argv[1]
    addr = sys.argv[2] if len(sys.argv) > 2 else None

    if mode == "send":
        # Demo: send a test game state
        bind = addr or "tcp://*:5555"
        sender = GameStateSender(bind)
        print("Sender bound to %s. Sending test message..." % bind)
        test_state = {
            "hand_id": "test123",
            "hero_cards": ["Ah", "Kd"],
            "board": ["Qh", "Jh", "Th"],
            "pot_bb": 10.0,
            "total_bb": 15.0,
            "street": "flop",
            "players": [],
            "available_actions": {"fold": True, "call": 5.0, "raise_to": 15.0},
        }
        # Send a few times (PUB may drop first messages before SUB connects)
        for _ in range(5):
            sender.socket.send_string(json.dumps(test_state))
            time.sleep(0.5)
        print("Sent test messages.")
        sender.close()

    elif mode == "recv":
        connect = addr or "tcp://localhost:5555"
        receiver = GameStateReceiver(connect)
        print("Receiver connected to %s. Waiting for messages..." % connect)
        try:
            while True:
                data = receiver.receive(timeout_ms=2000)
                if data:
                    print(json.dumps(data, indent=2))
                else:
                    print(".", end="", flush=True)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            receiver.close()
