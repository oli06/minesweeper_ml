#main and gui for our minesweeper game

from tkinter import *
import numpy as np
import math
import minesweeper as ms

board_size_x = 600
header_offset = 50
class MinesweeperUI():
    def __init__(self):
        self.window = Tk()
        self.window.resizable(False, False)
        self.window.title('Minesweeper')
        self.canvas = Canvas(self.window, width=board_size_x, height=board_size_x + header_offset)
        self.canvas.pack()
        # Input from user in form of clicks
        self.window.bind('<Button-1>', self.click)
        self.window.bind('<Button-2>', self.right_click)

        self.initialize_logic(9, 10)
        self.initialize_board()
        self.initalize_resizing_fields()

    def mainloop(self):
        self.window.mainloop()

    def initialize_logic(self, number_of_rows: int, number_of_mines:int):
        self.number_of_rows = number_of_rows
        self.number_of_mines = number_of_mines
        self.game = ms.Minesweeper(number_of_rows, number_of_mines)
        self.markers = np.zeros((number_of_rows, number_of_rows), dtype=np.bool_) #We need a field to hold track of markers

    def initalize_resizing_fields(self):
        self.game_size_entry = Entry(None) 
        self.canvas.create_window(400, 2 + board_size_x + (header_offset) / 4, window=self.game_size_entry)
        self.canvas.create_text(250, 2+ board_size_x + (header_offset) / 4, text="Spielgroeße", font="cmr 14")

        game_size_text = f"{self.number_of_rows}"
        self.game_size_entry.insert(0, game_size_text)

        self.game_mines_entry = Entry(self.window)
        self.canvas.create_window(400, 2+ board_size_x + 3 * (header_offset) / 4, window=self.game_mines_entry)
        self.canvas.create_text(250,2+ board_size_x + 3 * (header_offset) / 4, text="#Mines:", font="cmr 14",justify="right")
        mines_text = f"{self.number_of_mines}"
        self.game_mines_entry.insert(0, mines_text)

    def initialize_board(self):
        for i in range(self.number_of_rows+1): #columns
            self.canvas.create_line((i) * board_size_x / self.number_of_rows, 0, (i) * board_size_x / self.number_of_rows, board_size_x)

        for i in range(self.number_of_rows+1): #rows
            self.canvas.create_line(0, (i) * board_size_x / self.number_of_rows, board_size_x, (i) * board_size_x / self.number_of_rows)

        resetButton = self.canvas.create_rectangle(0, 2 + board_size_x, 200, board_size_x + header_offset - 4, fill="grey100", outline="grey60")
        resetButtonLabel = self.canvas.create_text(75, 2+ board_size_x + (header_offset-4) / 2, text="Neues Spiel", font="cmr 25 bold")
        self.canvas.tag_bind(resetButton, "<Button-1>", self.resetGame)
        self.canvas.tag_bind(resetButtonLabel, "<Button-1>", self.resetGame)
        
        self.drawScore()

    def drawScore(self):
        self.canvas.delete("score")
        text = f"Score: {self.game.unfolded}"
        self.canvas.create_text(550, board_size_x + (header_offset-4) / 2, font="cmr 15 bold", tags="score", fill="black", text=text)

    def resetGame(self, event):
        self.canvas.delete("all")
        game_size = self.game_size_entry.get()
        mines = self.game_mines_entry.get()
        self.initialize_logic(int(game_size), int(mines))
        self.initialize_board()
        self.initalize_resizing_fields()

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

        self.canvas.create_text(x + (board_size_x / self.number_of_rows) / 2, y + (board_size_x / self.number_of_rows) / 2, font="cmr 40 bold", fill=color, text=str(int(number)))


    def draw_marker(self, i,j):
        y,x = self.convert_logical_to_grid_position(i,j)

        self.canvas.create_text(x + (board_size_x / self.number_of_rows) / 2, y + (board_size_x / self.number_of_rows) / 2, font="cmr 40 bold", fill="grey40", text=str("x"), tags=f"marker_{i}_{j}")


    def remove_marker(self, i, j):
        self.canvas.delete(f"marker_{i}_{j}")


    def draw_mines(self):
        for ix, iy in np.ndindex(self.game.field_assignment.shape):
            if self.game.field_assignment[ix,iy] == math.inf:
                y,x = self.convert_logical_to_grid_position(ix,iy)
                self.canvas.create_text(x + (board_size_x / self.number_of_rows) / 2, y + (board_size_x / self.number_of_rows) / 2, font="cmr 40 bold", fill="black", text=str("B"))


    def convert_grid_to_logical_position(self, grid_position):
        return math.floor(grid_position[0] / board_size_x * self.number_of_rows), math.floor(grid_position[1] / board_size_x * self.number_of_rows)


    def convert_logical_to_grid_position(self, i,j):
        return (board_size_x / self.number_of_rows) * i, (board_size_x / self.number_of_rows) * j


    def click(self, event):
        grid_position = [event.x, event.y]
        print(event)
        print(event.widget)
        print(grid_position)
        if event.x < 0 or event.y < 0 or event.y > board_size_x:
            #clicked out of bounds / moved the window / reseted the game
            return

        j,i = self.convert_grid_to_logical_position(grid_position) #be careful: event.x and event.y axis are "switched"
        old_field = np.array(self.game.field) #TODO: unsauber programmiert. Bei jedem Click wird das Feld kopiert, um anschliessend veraenderungen festzustellen, um diese dann zu zeichnen
        if not self.game.unfold(i,j):
            #redraw ui
            new_fields = np.logical_xor(old_field, self.game.field)
            self.draw_active_fields(new_fields)
            self.drawScore()
        else:
            #game is lost or won
            if not self.game.is_game_won():
                self.draw_mines()
            else: 
                new_fields = np.logical_xor(old_field, self.game.field)
                self.draw_active_fields(new_fields)
                self.drawScore()

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