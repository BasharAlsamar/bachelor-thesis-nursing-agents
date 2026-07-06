import argparse
import socket
import struct
import time
from pathlib import Path

import cv2
import numpy as np


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    bytes_recd = 0
    while bytes_recd < size:
        chunk = sock.recv(min(65536, size - bytes_recd))
        if not chunk:
            raise ConnectionError("Verbindung vom Server geschlossen")
        chunks.append(chunk)
        bytes_recd += len(chunk)
    return b"".join(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="172.23.96.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--output-dir", default="chatbot/saved_frames")
    parser.add_argument("--max-frames", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Verbinde mit {args.host}:{args.port}...")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((args.host, args.port))
    print("Verbunden!")

    payload_size = struct.calcsize("Q")
    saved_count = 0

    try:
        while saved_count < args.max_frames:
            packed_size = recv_exact(client, payload_size)
            msg_size = struct.unpack("Q", packed_size)[0]

            # JPEG-Rohdaten empfangen
            frame_data = recv_exact(client, msg_size)

            # ✅ Auflösung prüfen ohne Qualitätsverlust
            np_data = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            if frame is None:
                print("⚠️ Ungültiger Frame, übersprungen")
                continue
            h, w = frame.shape[:2]

            # ✅ JPEG-Bytes direkt speichern (keine zweite Komprimierung!)
            timestamp_ms = int(time.time() * 1000)
            file_path = output_dir / f"frame_{saved_count + 1:02d}_{timestamp_ms}.jpg"
            file_path.write_bytes(frame_data)

            saved_count += 1
            print(f"[{saved_count}/{args.max_frames}] {w}x{h} → {file_path.name}")

        print(f"\n✅ Fertig! {saved_count} Frames gespeichert in: {output_dir}")

    except KeyboardInterrupt:
        print("Unterbrochen")
    except ConnectionError as exc:
        print(f"Verbindungsfehler: {exc}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
