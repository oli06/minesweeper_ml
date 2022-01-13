#main and gui for our minesweeper game

from tkinter import *
import numpy as np
import math
import minesweeper as ms

number_of_mines = 10
number_of_rows = 9

size_of_board = 603

class MinesweeperUI():
    def __init__(self):
        self.window = Tk()
        self.window.title('Minesweeper')
        self.canvas = Canvas(self.window, width=size_of_board, height=size_of_board)
        self.canvas.pack()
        # Input from user in form of clicks
        self.window.bind('<Button-1>', self.click)
        self.window.bind('<Button-2>', self.right_click)

        self.game = ms.Minesweeper(number_of_rows, number_of_mines)
        self.markers = np.zeros((number_of_rows, number_of_rows), dtype=np.bool_) #We need a field to hold track of markers
        self.initialize_board()

    def mainloop(self):
        self.window.mainloop()


    def initialize_board(self):
        for i in range(number_of_rows):
            self.canvas.create_line((i+1) * size_of_board / number_of_rows, 0, (i + 1) * size_of_board / number_of_rows, size_of_board)

        for i in range(number_of_rows+1):
            self.canvas.create_line(0, (i+1) * size_of_board / number_of_rows, size_of_board, (i + 1) * size_of_board / number_of_rows)


    def draw_active_fields(self, fields):
        for ix, iy in np.ndindex(fields.shape):
            if fields[ix, iy]:
                if self.markers[ix,iy]:
                    #first remove the marker
                    self.markers[ix,iy] = False
                    self.remove_marker(ix,iy)
                self.draw_number(ix, iy, self.game.field_assignment[ix, iy])


    def draw_number(self, i,j, number):
        y,x = self.convert_logical_to_grid_position(i,j)
        color = "black" #color for 8 mines as neighbors
        if number == 0:
            color = "blue"
        elif number == 1:
            color = "green"
        elif number == 2:
            color = "red"
        elif number == 3:
            color = "gray"
        elif number == 4:
            pass
        elif number == 5:
            pass
        elif number == 6:
            pass
        elif number == 7:
            pass

        self.canvas.create_text(x + (size_of_board / number_of_rows) / 2, y + (size_of_board / number_of_rows) / 2, font="cmr 40 bold", fill=color, text=str(int(number)))


    def draw_marker(self, i,j):
        y,x = self.convert_logical_to_grid_position(i,j)

        self.canvas.create_text(x + (size_of_board / number_of_rows) / 2, y + (size_of_board / number_of_rows) / 2, font="cmr 40 bold", fill="red", text=str("x"), tags=f"marker_{i}_{j}")


    def remove_marker(self, i, j):
        self.canvas.delete(f"marker_{i}_{j}")


    def draw_mines(self):
        for ix, iy in np.ndindex(self.game.field_assignment.shape):
            if self.game.field_assignment[ix,iy] == math.inf:
                y,x = self.convert_logical_to_grid_position(ix,iy)
                self.canvas.create_text(x + (size_of_board / number_of_rows) / 2, y + (size_of_board / number_of_rows) / 2, font="cmr 40 bold", fill="black", text=str("B"))


    def convert_grid_to_logical_position(self, grid_position):
        return math.floor(grid_position[0] / size_of_board * number_of_rows), math.floor(grid_position[1] / size_of_board * number_of_rows)


    def convert_logical_to_grid_position(self, i,j):
        return (size_of_board / number_of_rows) * i, (size_of_board / number_of_rows) * j


    def click(self, event):
        grid_position = [event.x, event.y]
        if event.x < 0 or event.y < 0:
            #clicked out of bounds / moved the window
            return

        j,i = self.convert_grid_to_logical_position(grid_position) #be careful: event.x and event.y axis are "switched"
        old_field = np.array(self.game.field) #TODO: unsauber programmiert. Bei jedem Click wird das Feld kopiert, um anschliessend veraenderungen festzustellen, um diese dann zu zeichnen
        if self.game.unfold(i,j):
            #redraw ui
            new_fields = np.logical_xor(old_field, self.game.field)
            print(new_fields)
            self.draw_active_fields(new_fields)
        else:
            #game is lost
            self.draw_mines()


    def right_click(self, event):
        grid_position = [event.x, event.y]
        if event.x < 0 or event.y < 0:
            #clicked out of bounds
            return 
        j,i = self.convert_grid_to_logical_position(grid_position) #be careful: event.x and event.y axis are "switched"
        if not self.game.field[i,j]:
            #marker or remove marker for this field
            if self.markers[i,j]:
                self.remove_marker(i,j)
            else:
                self.draw_marker(i,j)

            self.markers[i,j] = not self.markers[i,j]

game_instance = MinesweeperUI()
game_instance.mainloop()