import math as m


class Tree:
    def __init__(self, field, entropy):
        self.field = field
        self.connections = dict()
        self.entropy_decrease = entropy

    def __str__(self, nesting=0):
        res = f"\t" * nesting + self.field + ", Entropy Decrease: " + str(self.entropy_decrease) + "\n"
        for key, value in self.connections.items():
            res += "\t" * (nesting+1) + key + "\n"
            if isinstance(value, Tree):
                res += value.__str__(nesting + 2)
            elif isinstance(value, str):
                res += "\t" * (nesting+2) + value + "\n"
        return res

class DecisionTree:
    def __init__(self):
        self.tree = None

    def execute(self, data, output_field):
        """
        Calculate the decision tree using recursion.

        :param DataFrame data: pandas dataframe containing the data
        :param str output_field: name of the output field
        :return Tree:
        """
        self.output_field = output_field

        columns = list(data.columns)
        if len(columns) == 2:
            # Add leaves
            field = columns[0] if columns[0] != output_field else columns[1]
            values = data[field].unique()
            tree = Tree(field=field, entropy='No need to calculate')
            for value in values:
                leaf_value = data[data[field] == value][output_field].iloc[0]
                tree.connections[value] = leaf_value
            return tree
        else:
            best_field, best_entropy = self.get_best_field(data)
            tree = Tree(field=best_field, entropy=best_entropy)
            for value in data[best_field].unique():
                subset = self.create_subset(data, best_field, value).copy(deep=True)
                subset = subset.drop(best_field, axis=1)

                subtree = self.execute(subset, output_field)
                if all([not isinstance(value, Tree) for value in subtree.connections.values()]) and len(set(subtree.connections.values())) == 1:
                    tree.connections[value] = subtree.connections.values().__iter__().__next__()
                else:
                    tree.connections[value] = subtree
            return tree

    def get_best_field(self, data):
        """
        Returns the column having the best entropy and its entropy decrease.

        :param DataFrame data:
        :return:
        """
        search_fields = [field for field in data.columns if field != self.output_field]

        best_entropy_value = -1
        best_entropy_field = None
        data_entropy = self.calculate_data_entropy(data)
        for field in search_fields:
            field_entropy = self.calculate_field_entropy(data, field)
            field_entropy = data_entropy - field_entropy

            if field_entropy > best_entropy_value:
                best_entropy_value = field_entropy
                best_entropy_field = field
        return best_entropy_field, best_entropy_value

    def create_subset(self, data, field, value):
        """
        Creates subset of data where the column field is equal to value.

        :param DataFrame data:
        :param str field:
        :param str value:
        :return:
        """
        return data[data[field] == value]

    def calculate_field_entropy(self, data, field):
        """
        Calculate entropy for the field <field>.

        :param data:
        :param field:
        :return:
        """
        unique_values = data[field].unique()
        entropy = 0

        for value in unique_values:
            subset = self.create_subset(data, field, value)[self.output_field]
            count = subset.value_counts(normalize=True)
            value_entropy = 0
            for output in subset.unique():
                value_entropy += - (count[output] * m.log2(count[output]))
            entropy += value_entropy * (len(subset) / len(data))
        return entropy

    def calculate_data_entropy(self, data):
        """
        Calculate entropy for the data.
        :param data:
        :return:
        """
        unique_values = data[self.output_field].unique()
        count = data[self.output_field].value_counts(normalize=True)
        entropy = 0

        for value in unique_values:
            entropy += - (count[value] * m.log2(count[value]))
        return entropy



