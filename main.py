from utils import scheduler
if __name__ == "__main__":
    scheduler.start()
    print("Background schedulers started — press Ctrl+C to exit.")
    import time; [time.sleep(1) for _ in iter(int, 1)]
