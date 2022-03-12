from random import randint
import numpy as np


class BruteforceMinesweeperObject:
    def __init__(self, game_size_x, game_size_y):
        self.game_size_x = game_size_x
        self.game_size_y = game_size_y
        self.mines_for_sure = np.zeros((game_size_x, game_size_y))
        self.safe_for_sure = []
        self.danger_zone = []

    def __danger_zone_iteration(self, local_danger_zone):
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

    def __safe_spaces_of_danger_zones(self):
        for dgz in self.danger_zone:
            for dgz2 in self.danger_zone:
                if dgz != dgz2:
                    if len(dgz) < len(dgz2):
                        #nur wenn erste Liste echt kleiner ist kann sie vollstaendig in der anderen liste enthalten sein und es gibt ein safe_field
                        not_contained = False
                        for item in dgz:
                            if not item in dgz2:
                                not_contained = True
                                break
                        if not not_contained:
                            #alle felder, die nur in dgz2 sind, nicht aber in dgz sind safe_fields
                            for item in dgz2:
                                if not item in dgz:
                                    self.safe_for_sure.append(item)

    def __unfoldNeighboursOfZero(self, field: np.ndarray, field_assignment: np.ndarray):
        for i in range(0,self.game_size_x):
            for j in range(0, self.game_size_y):
                left = max(0, j-1)
                right = min(j + 1, self.game_size_x-1)
                top = max(0, i-1)
                bottom = min(i+1, self.game_size_y-1)
                neighbors = field_assignment[top:bottom+1, left:right+1]
                unfolded_neighbors = field[top:bottom+1, left:right+1]
                if field[i,j]:
                    if np.count_nonzero(unfolded_neighbors) == neighbors.shape[0] * neighbors.shape[1]:#wenn alle nachbarn schon aufgedeckt sind, dann ueberspringe dieses Feld
                        continue
                    if field_assignment[i,j] == 0: #ist ein Feld 0, dann sind alle nachbarn keine Minen
                        for o in range(left, right+1):
                            for p in range(top, bottom+1):
                                if not field[p, o] and not (p,o) in self.safe_for_sure:
                                    self.safe_for_sure.append((p,o))

    def __mines_for_sure_if_folded_equals_number(self, field: np.ndarray, field_assignment: np.ndarray):
        for i in range(0,self.game_size_x):
            for j in range(0, self.game_size_y):
                left = max(0, j-1)
                right = min(j + 1, self.game_size_x-1)
                top = max(0, i-1)
                bottom = min(i+1, self.game_size_y-1)
                neighbors = field_assignment[top:bottom+1, left:right+1]
                unfolded_neighbors = field[top:bottom+1, left:right+1]
                if field[i,j]:
                    if np.count_nonzero(unfolded_neighbors) == neighbors.shape[0] * neighbors.shape[1]:#wenn alle nachbarn schon aufgedeckt sind, dann ueberspringe dieses Feld
                        continue
                    value = field_assignment[i,j]
                    if value != 0 and value == np.count_nonzero(unfolded_neighbors==False):
                        #100% sicher ne bombe, weil alle  nicht aufgedeckter nachbarn gleich dem Wert des felds ist
                        for o in range(left, right+1):
                            for p in range(top, bottom+1):
                                if not field[p, o]:
                                    self.mines_for_sure[p,o] = 1


    def bruteforce_prediction(self, field: np.ndarray, field_assignment: np.ndarray):
        if len(self.safe_for_sure) > 0:
            move = self.safe_for_sure.pop()
            return move

        if len(self.safe_for_sure) > 0:
            return self.safe_for_sure.pop()

        self.__unfoldNeighboursOfZero(field, field_assignment)
        self.__mines_for_sure_if_folded_equals_number(field, field_assignment)
        
        if len(self.safe_for_sure) > 0:
            return self.safe_for_sure.pop()

        self.danger_zone = []
        for i in range(0,self.game_size_x):
            for j in range(0, self.game_size_y):
                if field[i,j]:
                    self.__danger_zone_generator(i,j, field, field_assignment)

        for i in range(0,self.game_size_x):
            for j in range(0, self.game_size_y):
                left = max(0, j-1)
                right = min(j + 1, self.game_size_x-1)
                top = max(0, i-1)
                bottom = min(i+1, self.game_size_y-1)
                neighbors = field_assignment[top:bottom+1, left:right+1]
                unfolded_neighbors = field[top:bottom+1, left:right+1]
                if field[i,j]:
                    value = field_assignment[i,j]
                    neighbor_mines = self.mines_for_sure[top:bottom+1, left:right+1]

                    if np.count_nonzero(unfolded_neighbors) == neighbors.shape[0] * neighbors.shape[1]:#wenn alle nachbarn schon aufgedeckt sind, dann ueberspringe dieses Feld
                        continue

                    if value != 0 and value == np.count_nonzero(neighbor_mines == 1): #wenn schon genug danger zones da sind (also genug bomben identifiziert als nachbarn)
                        for o in range(left, right+1):
                            for p in range(top, bottom+1):
                                if not field[p, o] and neighbor_mines[p-top, o-left] == 0 and not (p,o) in self.safe_for_sure:
                                    self.safe_for_sure.append((p,o))
                    
                        if len(self.safe_for_sure):
                            return self.safe_for_sure.pop()

                    field_danger_zones = self.__get_field_danger_zones(left, right, top, bottom) #alle dangerzones, die vollstaendig in der nachbar-range eines feldes i,j liegen

                    if len(field_danger_zones) - 1 == field_assignment[i,j]: #wir wissen, dass es felder gibt, die 100% keine minen sind
                        for o in range(left, right+1):
                            for p in range(top, bottom+1):
                                if (o != j or p != i) and not field[p, o]:
                                    counter = 0
                                    for dgz in field_danger_zones:
                                        if (p,o) in dgz:
                                            counter += 1
                                    
                                    if counter == 1 and not self.mines_for_sure[p,o] and not (p,o) in self.safe_for_sure:
                                        self.safe_for_sure.append((p,o))

        move = self.safe_for_sure.pop() if len(self.safe_for_sure) > 0 else None
        if move:
            self.danger_zone = [x for x in self.danger_zone if move not in x]
        return move

    def __get_field_danger_zones(self, left, right, top, bottom):
        field_danger_zones = []
        for dgz in self.danger_zone:
            is_contained = True
            for dgz_field in dgz:
                if not (left <= dgz_field[1] <= right and top <= dgz_field[0] <= bottom):
                    is_contained = False
                    break
                        
            if is_contained:
                field_danger_zones.append(dgz)

        return field_danger_zones

    def __danger_zone_generator(self, i,j, field, field_assignment):
        left = max(0, j-1)
        right = min(j + 1, self.game_size_x-1)
        top = max(0, i-1)
        bottom = min(i+1, self.game_size_y-1)

        local_danger_zone = []
        for o in range(left, right+1):
            for p in range(top, bottom+1):
                if not field[p, o] and not self.mines_for_sure[p,o]:
                    local_danger_zone.append([(p,o)])

        old_local_danger_zone = local_danger_zone   
        new_local_danger_zone = self.__danger_zone_iteration(old_local_danger_zone)
        while not new_local_danger_zone is None:
            old_local_danger_zone = new_local_danger_zone
            new_local_danger_zone = self.__danger_zone_iteration(old_local_danger_zone)

        remove_dgz_pls = []
        for dgz in old_local_danger_zone:
            if len(dgz) == 1:
                value = field_assignment[i,j]
                unfolded_neighbors = field[top:bottom+1, left:right+1]
                if value < np.count_nonzero(unfolded_neighbors==False):
                    remove_dgz_pls.append(dgz)

        for rm_dgz in remove_dgz_pls:
            old_local_danger_zone.remove(rm_dgz)

        self.danger_zone.extend(old_local_danger_zone)
        remove_idx = []
        for i in range(0, len(self.danger_zone)):
            for j in range(i, len(self.danger_zone)):
                if i != j:
                    if self.danger_zone[i] == self.danger_zone[j]:
                        remove_idx.append(j)

        for i in sorted(remove_idx,reverse=True):
            self.danger_zone.pop(i)


"""

arr = np.array([[ 0., 1,  1,  2, 1,  1],
       [ 1,  2,  math.inf,  2,  math.inf,  2.],
       [ math.inf,  3,  3,  4,  3,  math.inf],
       [1,  2,  math.inf,  math.inf,  2,  1.],
       [1 ,  2,  2,  2, 2,  1.],
       [math.inf,  1.,  0,  0.,  1.,  math.inf]
       ])
f_arr = np.array([
     [False, False, False, False, False, False],
      [False, False, False, True, False, False],
       [False, False, False, True, False, False],
        [False, True, False, False, True, False],
         [False, True, True, True, True, False],
          [False, True, True, True, True, False]
      ])
pretty_print(f_arr, arr, 6)
safe_guess = bruteforce_prediction(f_arr, arr, 6)
while not safe_guess == None:
    print(safe_guess)
    print(safe_for_sure)
    f_arr[safe_guess[0], safe_guess[1]] = True
    pretty_print(f_arr, arr,6)
    safe_guess = bruteforce_prediction(f_arr, arr, 6)











bruteforce_won_games = []
not_solved = []
lost_games = []
for i in range(100):
    safe_for_sure = []
    mines_for_sure = np.zeros((9,9))
    danger_zone = []
    game = initial_bruteforce_unfolding(minesweeper.Minesweeper(9, 10))
    move = bruteforce_prediction(game.field, game.field_assignment, 9)

    done = False
    while move:
        if game.unfold(move[0], move[1]):
            done = True
            if game.is_game_won():
                for i in range(0,9):
                    for j in range(0,9):
                        if game.field[i,j]:
                            if game.field_assignment[i,j] == math.inf:
                                print("fehler")
                        if not game.field[i,j]:
                            if game.field_assignment[i,j] != math.inf:
                                print("fehler too")
                bruteforce_won_games.append(game)
            else:
                lost_games.append((game, move, mines_for_sure, safe_for_sure, danger_zone))
            break
        move = bruteforce_prediction(game.field, game.field_assignment, 9)
    if not done:
        not_solved.append(game)

print("done")
f_arr = lost_games[0][0].field
arr = lost_games[0][0].field_assignment
danger_zone = []
mines_for_sure = lost_games[0][2]
safe_for_sure = lost_games[0][3]
pretty_print(f_arr, arr,9)

safe_guess = bruteforce_prediction(f_arr, arr, 9)
while not safe_guess == None:
    print(safe_guess)
    print(safe_for_sure)
    f_arr[safe_guess[0], safe_guess[1]] = True
    pretty_print(f_arr, arr,9)
    safe_guess = bruteforce_prediction(f_arr, arr, 9)

arr = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  1., math.inf,  1.],
       [ 1.,  1.,  0.,  0.,  0.,  1.,  2.,  2.,  1.],
       [math.inf,  1.,  0.,  1.,  1.,  2., math.inf,  1.,  0.],
       [ 2.,  2.,  0.,  1., math.inf,  2.,  2.,  2.,  1.],
       [math.inf,  2.,  1.,  2.,  1.,  1.,  1., math.inf,  1.],
       [ 1.,  2., math.inf,  1.,  0.,  0.,  2.,  2.,  2.],
       [ 0.,  2.,  2.,  3.,  1.,  1.,  1., math.inf,  1.],
       [ 0.,  1., math.inf,  2., math.inf,  1.,  1.,  1.,  1.]])
f_arr = np.array([[ True,  True,  True,  True,  True,  True,  True, False, False],
       [ True,  True,  True,  True,  True,  True,  True, False,  True],
       [ True,  True,  True,  True,  True,  True,  True,  True,  True],
       [False,  True,  True,  True,  True,  True, False,  True,  True],
       [ True,  True,  True,  True, False, False, False,  True,  True],
       [False,  True,  True,  True, False, False, False, False, False],
       [False, False, False, False, False, False, False, False, False],
       [False, False, False, False, False, False, False, False, False],
       [False, False, False, False, False, False, False, False, False]])
mines_for_sure[3,0] = 1
mines_for_sure[3,6] = 1
pretty_print(f_arr, arr, 9)
safe_guess = bruteforce_prediction(f_arr, arr, 9)
while not safe_guess == None:
    print(safe_guess)
    print(safe_for_sure)
    f_arr[safe_guess[0], safe_guess[1]] = True
    pretty_print(f_arr, arr,9)
    safe_guess = bruteforce_prediction(f_arr, arr, 9)


"""