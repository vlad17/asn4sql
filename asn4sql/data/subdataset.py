"""
Returns a pytorch dataset with some of the fields removed.
"""

import torchtext.data


class Subdataset(torchtext.data.Dataset):
    """
    Removes the specified columns from the dataset.
    """

    def __init__(self, dataset, columns_to_drop):
        all_columns = list(dataset.fields.keys())
        fields = {
            name: field
            for name, field in dataset.fields.items()
            if name not in columns_to_drop
        }
        columns_to_keep = list(set(all_columns) - set(columns_to_drop))
        examples = [
            torchtext.data.Example.fromlist(
                [getattr(ex, col)
                 for col in columns_to_keep], [(col, dataset.fields[col])
                                               for col in columns_to_keep])
            for ex in dataset.examples
        ]
        super().__init__(examples, fields)
