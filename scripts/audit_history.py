import subprocess
from collections import Counter
from datetime import date, datetime, time
from zoneinfo import ZoneInfo


def main() -> None:
    history = subprocess.check_output(
        ["git", "log", "--reverse", "--format=%H%x09%aI%x09%cI%x09%s"], text=True
    ).splitlines()
    if len(history) != 52:
        raise ValueError(f"Expected 52 commits, found {len(history)}.")
    start, end = date(2026, 4, 7), date(2026, 10, 1)
    zone = ZoneInfo("Europe/Warsaw")
    timestamps = []
    weeks: Counter[int] = Counter()
    days: dict[int, set[date]] = {}
    for line in history:
        commit, authored, committed, message = line.split("\t", maxsplit=3)
        if authored != committed:
            raise ValueError(f"Author and committer timestamps differ for {commit}.")
        timestamp = datetime.fromisoformat(authored).astimezone(zone)
        if not start <= timestamp.date() <= end or timestamp > datetime.now(zone):
            raise ValueError(f"Timestamp outside the requested range: {timestamp}.")
        if not time(13) <= timestamp.time() <= time(23, 45):
            raise ValueError(f"Timestamp outside afternoon/evening hours: {timestamp}.")
        if len(message) > 72 or not message.endswith(".") or "\n" in message:
            raise ValueError(f"Commit message is not a short sentence: {message}.")
        changed = subprocess.check_output(
            ["git", "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit],
            text=True,
        )
        if not changed.strip():
            raise ValueError(f"Empty commit: {commit}.")
        week = (timestamp.date() - start).days // 7
        weeks[week] += 1
        days.setdefault(week, set()).add(timestamp.date())
        timestamps.append(timestamp)
    if timestamps != sorted(timestamps) or len(set(timestamps)) != 52:
        raise ValueError("Commit timestamps must increase without duplicates.")
    if weeks != Counter({week: 2 for week in range(26)}) or any(
        len(day) != 2 for day in days.values()
    ):
        raise ValueError("Each weekly window must contain two distinct commit dates.")
    weekends = sum(timestamp.weekday() >= 5 for timestamp in timestamps)
    if weekends <= len(timestamps) / 2:
        raise ValueError("The timeline does not favor weekends.")
    if len({timestamp.time() for timestamp in timestamps}) < 40:
        raise ValueError("Commit times lack variation.")
    print(f"52 commits; 26 weekly windows; {weekends} weekend commits.")
    print(f"First: {timestamps[0].isoformat()}; last: {timestamps[-1].isoformat()}.")


if __name__ == "__main__":
    main()
