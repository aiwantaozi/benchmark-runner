"""Success-rate policy shared by benchmark search and reporting."""

# Tolerate occasional request failures without accepting an overloaded point.
SUCCESS_FLOOR = 0.95
