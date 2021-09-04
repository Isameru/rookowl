""" Contains `PieceDataset` class and related utilities to cut the chessboard photograph into 64
    images for each chessboard piece or square.
"""

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-

PIECE_TEXT_MAP = {
    ".": "Empty Square (Puste Pole)",
    "P": "White Pawn (Biały Pionek)",
    "p": "Black Pawn (Czarny Pionek)",
    "B": "White Bishop (Biały Goniec)",
    "b": "Black Bishop (Czarny Goniec)",
    "N": "White Knight (Biały Skoczek)",
    "n": "Black Knight (Czarny Skoczek)",
    "R": "White Rook (Biała Wieża)",
    "r": "Black Rook (Czarna Wieża)",
    "Q": "White Queen (Biały Hetman)",
    "q": "Black Queen (Czarny Hetman)",
    "K": "White King (Biały Król)",
    "k": "Black King (Czarny Król)",
}

# A piece code is an interger user as a one-hot class for piece detection from an image.
PIECE_CODE_MAP = {
    ".": 0,
    "P": 1,
    "p": 2,
    "B": 3,
    "b": 4,
    "N": 5,
    "n": 6,
    "R": 7,
    "r": 8,
    "Q": 9,
    "q": 10,
    "K": 11,
    "k": 12,
}

PIECE_CODE_COUNT = len(PIECE_CODE_MAP)

# =≡=-=♔=-=≡=-=♕=-=≡=-=♖=-=≡=-=♗=-=≡=-=♘=-=≡=-=♙=-=≡=-=♚=-=≡=-=♛=-=≡=-=♜=-=≡=-=♝=-=≡=-=♞=-=≡=-=♟︎=-
