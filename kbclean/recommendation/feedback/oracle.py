class Oracle:
    def __init__(self, dataset):
        self.dataset = dataset

    def answer(self, col, row_index):
        return int(self.dataset.dirty_df[col][row_index] == self.dataset.clean_df[col][row_index])

    def get_pairs(self, col, row_index):
        return self.dataset.dirty_df[col][row_index], self.dataset.clean_df[col][row_index]
