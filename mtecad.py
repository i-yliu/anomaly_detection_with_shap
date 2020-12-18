class data_flow(Sequence):
    def __init__(self, filenames, batch_size, grayscale):
        self.filenames = filenames
        self.batch_size = batch_size
        self.grayscale = grayscale

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array([read_img(filename, self.grayscale) for filename in batch_x])

        batch_x = batch_x / 255.
        return batch_x, batch_x

# data
