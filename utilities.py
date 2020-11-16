def from_board_to_fne(board):
    def get_piece(piece):
        return {
            ".": False
        }.get(piece, piece)

    fne_board = []
    # 'for loop' until the shape is defined
    for row in board.splitlines():
        fne_row = ""
        empty_spaces = 0
        for piece in row:
            piece = get_piece(piece)
            if piece:
                if empty_spaces:
                    fne_row += str(empty_spaces)
                    fne_row += piece
                else:
                    fne_row += piece

                empty_spaces = 0
            else:
                empty_spaces += 1

        if empty_spaces:
            fne_row += str(empty_spaces)

        fne_board.append(fne_row)

    return "/".join(fne_board)[1:]


board = """
r.bqkb.r
pppp.Qpp
..n..n..
....p...
..B.P...
........
PPPP.PPP
RNB.K.NR
"""

print(from_board_to_fne(board))
