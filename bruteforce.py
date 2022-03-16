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

    def __unfoldNeighboursOfZero(self, field: np.ndarray, field_assignment: np.ndarray):
        '''
        Iterates over field_assignment and adds every field to self.save_for_sure if it has a zero-neighbour.
        
                Parameters:
                        field (array): the field array
                        field_assignment (array): the field_assignment array

                Returns:
                        void
        '''
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
        '''
        Iterates over field_assignment and adds every field to self.mines_for_sure if for a neighbours-value is equal to folded neighbours.
        
                Parameters:
                        field (array): the field array
                        field_assignment (array): the field_assignment array

                Returns:
                        void
        '''
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
        '''
        For a given 2D-range of fields all danger zones inside the range are returned
                Parameters:
                        left (int): left
                        right (int): right
                        top (int): top
                        bottom (int): bottom

                Returns:
                        Array of danger zones for the 2D-range
        '''
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
        '''
        Generates all danger zones for a given field (i,j) and stores them in self.danger_zone. 
        If self.danger_zone already contains a generated danger_zone, these are merged.

                Parameters:
                        i (int): y-coordinate of the field
                        j (int): x-coordinate of the field
                        field (array): the field array
                        field_assignment (array): the field_assignment array

                Returns:
                        void
        '''
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