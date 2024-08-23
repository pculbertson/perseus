def rank_print(msg: str, rank: int = 0) -> None:
    """Prints only if rank is 0."""
    if rank == 0:
        print(msg)
