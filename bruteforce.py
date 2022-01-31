from xmlrpc.client import Boolean
import numpy as np
import math
from collections import OrderedDict

mines_for_sure = np.zeros((9,9))
safe_for_sure = []
danger_zone = []


def danger_zone_iteration(local_danger_zone):
    for ldz_list in local_danger_zone:
            for ldz_field in ldz_list:
                for ldz2_list in local_danger_zone:
                    if ldz_list != ldz2_list:
                        for ldz2_field in ldz2_list:
                            if ldz_field != ldz2_field:
                                diff_first = abs(ldz_field[0] - ldz2_field[0])
                                diff_second = abs(ldz_field[1] - ldz2_field[1])
                                if np.bitwise_xor(diff_first, diff_second) == 1:
                                    new_danger_zone = ldz_list + ldz2_list
                                    local_danger_zone.append(new_danger_zone)
                                    local_danger_zone.remove(ldz_list)
                                    local_danger_zone.remove(ldz2_list)
                                    return local_danger_zone

    return None

def print_game(field, field_assignment, game_size):
    for i in range(0,game_size-1):
        print_str = " "
        for j in range(0,game_size-1):
            if mines_for_sure[i,j]:
                print_str += " B "
            else: 
                print_str += " U " if not field[i,j] else " " + str(int(field_assignment[i,j])) + " "
        print(print_str)

def bruteforce_prediction(field: np.ndarray, field_assignment: np.ndarray, game_size: int):
    global danger_zone
    if len(safe_for_sure) > 0:
        move = safe_for_sure.pop()
        danger_zone = [x for x in danger_zone if move not in x]
        return move

    for i in range(0,game_size-1):
        for j in range(0, game_size-1):
            if field[i,j]:
                danger_zone_generator(i,j, field, field_assignment, game_size)

    for i in range(0,game_size-1):
        for j in range(0, game_size-1):
            left = max(0, j-1)
            right = min(j + 1, game_size-1)
            top = max(0, i-1)
            bottom = min(i+1, game_size-1)
            neighbors = field_assignment[top:bottom+1, left:right+1]
            unfolded_neighbors = field[top:bottom+1, left:right+1]
            if field[i,j]:
                value = field_assignment[i,j]
                neighbor_mines = mines_for_sure[top:bottom+1, left:right+1]
                if value != 0 and value == np.count_nonzero(unfolded_neighbors==False):
                    #100% sicher ne bombe, weil alle  nicht aufgedeckter nachbarn gleich dem Wert des felds ist
                    for o in range(left, right+1):
                        for p in range(top, bottom+1):
                            if not field[p, o]:
                                mines_for_sure[p,o] = 1

                if value != 0 and value == np.count_nonzero(neighbor_mines == 1): #wenn schon genug danger zones da sind (also genug bomben identifiziert als nachbarn)
                    for o in range(left, right+1):
                        for p in range(top, bottom+1):
                            if not field[p, o] and neighbor_mines[p-top, o-left] == 0 and not (p,o) in safe_for_sure:
                                safe_for_sure.append((p,o))
                                    
                field_danger_zones = []
                for dgz in danger_zone:
                    is_contained = True
                    for dgz_field in dgz:
                        if not (left <= dgz_field[1] <= right and top <= dgz_field[0] <= bottom):
                            is_contained = False
                            break
                    
                    if is_contained:
                        field_danger_zones.append(dgz)

                neighbors_unfolded = field[top:bottom+1, left:right+1]
                if np.count_nonzero(neighbors_unfolded == True) == neighbors.shape[0] * neighbors.shape[1]:#wenn alle nachbarn schon aufgedeckt sind, dann ueberspringe
                    break

                if len(field_danger_zones) - 1 == field_assignment[i,j]: #wir wissen, dass es felder gibt, die 100% keine minen sind
                    for o in range(left, right+1):
                        for p in range(top, bottom+1):
                            print(f"nachbar {(p,o)} von feld {(i,j)}")
                            if (o != j or p != i) and not field[p, o]:
                                counter = 0
                                for dgz in field_danger_zones:
                                    if (p,o) in dgz:
                                        counter += 1
                                
                                if counter == 1 and not mines_for_sure[p,o] and not (p,o) in safe_for_sure:
                                    safe_for_sure.append((p,o))

    move = safe_for_sure.pop() if len(safe_for_sure) > 0 else None
    if move:
        danger_zone = [x for x in danger_zone if move not in x]
    return move

def danger_zone_generator(i,j, field, field_assignment, game_size):
    global danger_zone
    left = max(0, j-1)
    right = min(j + 1, game_size-1)
    top = max(0, i-1)
    bottom = min(i+1, game_size-1)

    local_danger_zone = []
    for o in range(left, right+1):
        for p in range(top, bottom+1):
            if not field[p, o]:
                local_danger_zone.append([(p,o)])

    old_local_danger_zone = local_danger_zone   
    new_local_danger_zone = danger_zone_iteration(old_local_danger_zone)
    while not new_local_danger_zone is None:
        old_local_danger_zone = new_local_danger_zone
        new_local_danger_zone = danger_zone_iteration(old_local_danger_zone)

    #for dgz in old_local_danger_zone:
    #    if len(dgz) == 1:
    #        mines_for_sure[dgz[0][0], dgz[0][1]] = 1
    remove_dgz_pls = []
    for dgz in old_local_danger_zone:
        if len(dgz) == 1:
            value = field_assignment[i,j]
            unfolded_neighbors = field[top:bottom+1, left:right+1]
            if value < np.count_nonzero(unfolded_neighbors==False):
                remove_dgz_pls.append(dgz)

    for rm_dgz in remove_dgz_pls:
        old_local_danger_zone.remove(rm_dgz)

    danger_zone.extend(old_local_danger_zone)
    remove_idx = []
    for i in range(0, len(danger_zone)):
        for j in range(i, len(danger_zone)):
            if i != j:
                if danger_zone[i] == danger_zone[j]:
                    remove_idx.append(j)

    for i in sorted(remove_idx,reverse=True):
        danger_zone.pop(i)
        


arr = np.array([[ 1,  1,  2, math.inf,  1,  0,  1, math.inf,  1],
 [ 1, math.inf,  2,  1,  1,  1,  2,  2,  1],
 [ 2,  2,  2,  1,  1,  1, math.inf,  1,  0],
 [math.inf,  1,  1, math.inf,  1,  2,  2,  2,  0],
 [ 1,  1,  1,  1,  1,  1, math.inf,  1,  0],
 [ 1,  1,  1,  0,  0,  1,  1,  1,  0],
 [ 1, math.inf,  1,  0,  0,  1,  1,  2,  1],
 [ 1,  1,  1,  0,  0,  1, math.inf,  2, math.inf],
 [ 0,  0,  0,  0,  0,  1,  1,  2,  1]])
 
f_arr = np.zeros((9,9), dtype=Boolean)
f_arr[0,0] = True
f_arr[0,1] = True
f_arr[1,0] = True
safe_guess = bruteforce_prediction(f_arr, arr, 9)
while not safe_guess == None:
    print(safe_guess)
    print(safe_for_sure)
    f_arr[safe_guess[0], safe_guess[1]] = True
    print_game(f_arr, arr, 9)
    safe_guess = bruteforce_prediction(f_arr, arr, 9)

f_arr[3,1] = True
safe_guess = bruteforce_prediction(f_arr, arr, 9)
while not safe_guess == None:
    print(safe_guess)
    print(safe_for_sure)
    f_arr[safe_guess[0], safe_guess[1]] = True
    print_game(f_arr, arr,9)
    safe_guess = bruteforce_prediction(f_arr, arr, 9)