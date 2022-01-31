import numpy as np

mines_for_sure = np.zeros((9,9))
safe_for_sure = []

def bruteforce_prediction(field: np.ndarray, field_assignment: np.ndarray, game_size: int):
    if len(safe_for_sure) > 0:
        return safe_for_sure.pop()

    for i,j in field:
        left = max(0, j-1)
        right = min(j + 1, game_size-1)
        top = max(0, i-1)
        bottom = min(i+1, game_size-1)
        neighbors = field_assignment[left:right,top:bottom]
        if field[i,j]:
            value = field_assignment[i,j]
            neighbor_mines = mines_for_sure[left:right, top:bottom]
            if value == np.count_nonzero(neighbor_mines == 1):
                safe_for_sure.append(neighbors[not neighbor_mines])

        if not field[i,j]: #falls noch nicht aufgedeckt
            #falls irgendeiner der nachbarn eine 0 ist, dann ist es def. keine Bombe
            neighbors = field_assignment[left:right,top:bottom]
            

    return None
