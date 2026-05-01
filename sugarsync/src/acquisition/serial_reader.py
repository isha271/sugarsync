"""
SugarSync — Serial Signal Acquisition
Reads NIR-PPG data from Arduino over USB serial and saves to Excel.

Usage:
    python src/acquisition/serial_reader.py --port COM3 --duration 120
    python src/acquisition/serial_reader.py --list-ports
"""

import argparse
import csv
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import pandas as pd
import serial
import serial.tools.list_ports
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from src.utils.logger import get_logger
from src.utils.config import load_config

log     = get_logger(__name__)
console = Console()
cfg     = load_config()


# ── Port Discovery ─────────────────────────────────────────────────────────

def list_ports() -> None:
    """Print all available serial ports."""
    ports = serial.tools.list_ports.comports()
    if not ports:
        console.print("[yellow]No serial ports found.[/yellow]")
        return
    table = Table(title="Available Serial Ports")
    table.add_column("Port",        style="cyan")
    table.add_column("Description", style="white")
    table.add_column("HWID",        style="dim")
    for p in ports:
        table.add_row(p.device, p.description, p.hwid)
    console.print(table)


# ── Real-time Stats Panel ──────────────────────────────────────────────────

def _make_panel(n_samples: int, hr: float, sqi: float, elapsed: float) -> Panel:
    body = (
        f"[bold cyan]Samples:[/bold cyan]  {n_samples}\n"
        f"[bold green]Heart Rate:[/bold green] {hr:.1f} bpm\n"
        f"[bold yellow]SQI:[/bold yellow]       {sqi:.3f}\n"
        f"[dim]Elapsed:   {elapsed:.1f}s[/dim]"
    )
    return Panel(body, title="[bold]SugarSync — Live Acquisition[/bold]", border_style="cyan")


# ── Main Acquisition Loop ──────────────────────────────────────────────────

def acquire(
    port: str,
    baud: int,
    duration_s: int,
    output_path: str,
    glucometer_value: float | None = None,
) -> pd.DataFrame:
    """
    Stream NIR-PPG from Arduino, save raw CSV + Excel.

    Parameters
    ----------
    port             : Serial port (e.g. 'COM3' or '/dev/ttyUSB0')
    baud             : Baud rate (default 115200)
    duration_s       : Recording duration in seconds
    output_path      : Where to save the .xlsx file
    glucometer_value : Optional ground-truth glucose reading (mg/dL)

    Returns
    -------
    pd.DataFrame with columns [timestamp_ms, nir_adc, red_adc]
    """
    rows: list[dict] = []
    nir_window: deque = deque(maxlen=100)   # 1s @ 100Hz for HR estimation
    last_peak_ts: float = 0.0
    hr: float = 0.0
    sqi_val: float = 0.0
    peak_in: bool = False
    start_time = time.time()

    log.info(f"Opening {port} @ {baud} baud …")

    try:
        ser = serial.Serial(port, baud, timeout=2)
    except serial.SerialException as exc:
        log.error(f"Could not open {port}: {exc}")
        sys.exit(1)

    time.sleep(2)   # wait for Arduino reset
    ser.reset_input_buffer()

    with Live(_make_panel(0, 0.0, 0.0, 0.0), refresh_per_second=4) as live:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration_s:
                break

            try:
                raw = ser.readline().decode("utf-8", errors="ignore").strip()
            except serial.SerialException:
                log.warning("Serial read error — retrying …")
                continue

            if not raw or raw.startswith("timestamp"):
                continue  # skip header line

            parts = raw.split(",")
            if len(parts) != 3:
                continue

            try:
                ts, nir, red = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue

            rows.append({"timestamp_ms": ts, "nir_adc": nir, "red_adc": red})
            nir_window.append(nir)

            # ── Real-time HR estimation ──────────────────────────────────
            if len(nir_window) == 100:
                mean_nir   = sum(nir_window) / 100
                threshold  = mean_nir * 1.08
                if not peak_in and nir > threshold:
                    peak_in   = True
                    now_s     = time.time()
                    dt        = now_s - last_peak_ts
                    if 0.3 < dt < 1.5:
                        hr = 60.0 / dt
                    last_peak_ts = now_s
                elif peak_in and nir < mean_nir:
                    peak_in = False

                # Simple SQI proxy: coefficient of variation of IBI
                # (real SQI computed properly in preprocessing)
                sqi_val = min(1.0, hr / 100.0) if hr > 0 else 0.0

            live.update(_make_panel(len(rows), hr, sqi_val, elapsed))

    ser.close()

    df = pd.DataFrame(rows)
    if glucometer_value is not None:
        df["glucometer_mg_dl"] = glucometer_value

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_excel(output_path, index=False)
    df.to_csv(output_path.with_suffix(".csv"), index=False)

    console.print(
        f"\n[bold green]✓ Saved {len(df)} samples → {output_path}[/bold green]"
    )
    log.info(f"Saved {len(df)} samples to {output_path}")
    return df


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SugarSync — NIR-PPG Serial Acquisition"
    )
    parser.add_argument("--list-ports", action="store_true",
                        help="List available serial ports and exit")
    parser.add_argument("--port",     default=cfg["acquisition"]["default_port"],
                        help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud",     type=int,
                        default=cfg["acquisition"]["baud_rate"])
    parser.add_argument("--duration", type=int, default=120,
                        help="Recording duration in seconds")
    parser.add_argument("--output",   default="data/raw/session.xlsx",
                        help="Output .xlsx file path")
    parser.add_argument("--glucose",  type=float, default=None,
                        help="Ground-truth glucometer reading (mg/dL)")

    args = parser.parse_args()

    if args.list_ports:
        list_ports()
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output if args.output != "data/raw/session.xlsx" \
        else f"data/raw/session_{timestamp}.xlsx"

    if args.glucose is not None:
        console.print(f"[cyan]Ground-truth glucose: {args.glucose} mg/dL[/cyan]")

    acquire(
        port=args.port,
        baud=args.baud,
        duration_s=args.duration,
        output_path=out,
        glucometer_value=args.glucose,
    )


if __name__ == "__main__":
    main()
