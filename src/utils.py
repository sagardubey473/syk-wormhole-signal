"""Utility functions for the wormhole research project."""

import numpy as np
import os
import time


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def timer(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


class ProgressTracker:
    """Simple progress tracker for long computations."""

    def __init__(self, total, desc=""):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()

    def update(self, n=1):
        self.current += n
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0
        print(f"\r{self.desc}: {self.current}/{self.total} "
              f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)",
              end='', flush=True)
        if self.current >= self.total:
            print()  # newline at end
