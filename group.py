import random
from collections import defaultdict

import torch
from torch import nn, softmax

random.seed(0)
torch.random.manual_seed(0)
torch.autograd.set_detect_anomaly(True)


def make_group(
    count_group: int, count_person_per_group: int
) -> tuple[dict[int, list[int]], dict[int, int]]:
    count_all = count_group * count_person_per_group
    group_mapping = defaultdict(list)
    rever_group_mapping = {}
    for i in range(count_all):
        group = i % count_group
        group_mapping[group].append(i)
        rever_group_mapping[i] = group

    return group_mapping, rever_group_mapping


def get_conflict_table(groups: dict[int, list[int]], count_conflict: int = 600):
    count_group = len(groups)
    count_person_per_group = len(groups[0])
    count_all = count_group * count_person_per_group

    conflict_table = torch.zeros((count_all, count_all), dtype=torch.float32)
    while conflict_table.sum() < count_conflict:
        group1 = random.randint(0, count_group - 1)
        group2 = random.randint(0, count_group - 1)
        if group1 != group2:
            person1 = random.choice(groups[group1])
            person2 = random.choice(groups[group2])
            conflict_table[person1, person2] = 1

    conflict_table = conflict_table.clone().detach().requires_grad_(True)
    return conflict_table


def get_bigrams(vector: list[int]):
    return [(vector[i], vector[i + 1]) for i in range(len(vector) - 1)]


def _get_duplicate_loss(persons, probabilities):
    probabilities = probabilities + 1
    unique_persons, count = torch.unique(persons, return_counts=True)
    duplicate_persons = unique_persons[count > 1]
    duplicate_indexes = torch.where(
        persons.unsqueeze(0) == duplicate_persons.unsqueeze(1)
    )
    duplicate_probabilities = probabilities[duplicate_indexes[1]]
    loss = duplicate_probabilities.prod() * count.prod()
    return loss


def get_neighbors(i: int, chair_size: tuple[int, int]) -> set[int]:
    x, y = divmod(i, chair_size[1])
    neighbors = set()
    if x > 0:
        neighbors.add(i - chair_size[1])
    if x < chair_size[0] - 1:
        neighbors.add(i + chair_size[1])
    if y > 0:
        neighbors.add(i - 1)
    if y < chair_size[1] - 1:
        neighbors.add(i + 1)
    return neighbors


def _get_conflict_loss(persons, probabilities, conflict_table, chair_size):
    loss = 0
    for i, person in enumerate(persons):
        neighbors = get_neighbors(person.item(), chair_size)
        for neighbor in neighbors:
            if conflict_table[person, persons[neighbor]]:
                loss += (probabilities[i] + 1) * (probabilities[neighbor] + 1)
    return loss


def dfs(position, persons, visited, subgroup, group_label, chair_size):
    stack = [position]
    while stack:
        pos = stack.pop()
        if pos not in visited:
            visited.add(pos)
            subgroup.append(pos)
            for neighbor in get_neighbors(pos, chair_size):
                if persons[neighbor] == group_label and neighbor not in visited:
                    stack.append(neighbor)


def find_all_subgroups(persons, reverse_group_mapping, chair_size):
    visited = set()
    subgroups = defaultdict(list)

    for i, person in enumerate(persons):
        group_label = reverse_group_mapping[person.item()]
        if i not in visited:
            subgroup = []
            dfs(i, persons, visited, subgroup, group_label, chair_size)
            subgroups[group_label].append(subgroup)

    return subgroups


def _get_group_loss(persons, probabilities, reverse_group_mapping, chair_size):
    group_loss = defaultdict(lambda: defaultdict(int))
    loss = 0
    for i, person in enumerate(persons):
        group = reverse_group_mapping[person.item()]
        if group_loss[group][person.item()]:
            loss += (probabilities[i] + 1) * group_loss[group][person.item()]
        neighbors = get_neighbors(i, chair_size)

    return loss


def get_loss(
    output: torch.Tensor,
    reverse_group_mapping: dict[int, int],
    conflict_table: torch.Tensor,
    chair_size: tuple[int, int],
):
    o = softmax(output, dim=1).max(dim=1)
    persons = o.indices
    probabilities = o.values

    loss = _get_duplicate_loss(persons, probabilities)
    loss += _get_conflict_loss(persons, probabilities, conflict_table, chair_size)
    # loss += _get_group_loss(persons, probabilities, reverse_group_mapping, chair_size)
    return loss


class ConflictModule(nn.Module):
    def __init__(self, chair_size: tuple[int, int]):
        super(ConflictModule, self).__init__()
        self.chair_size = chair_size
        self.count_all = self.chair_size[0] * self.chair_size[1]
        self.l1 = nn.Linear(self.count_all, self.count_all, bias=False)
        self.l2 = nn.Linear(self.count_all, self.count_all, bias=False)

    def forward(self, conflict_table: torch.Tensor):
        output = torch.relu(self.l1(conflict_table))
        output = self.l2(output)
        # output = self.l1(conflict_table)
        return output

    def train_model(
        self, conflict_table: torch.Tensor, reverse_group_mapping: dict[int, int]
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        input_vector = torch.eye(self.count_all)
        output = input_vector
        for i in range(100):
            optimizer.zero_grad()
            output = self.forward(input_vector)
            loss = get_loss(
                output, reverse_group_mapping, conflict_table, self.chair_size
            )
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"{i} loss:", loss.item())
                probabilities = softmax(output, dim=1).max(dim=1)

                print_2d(probabilities.indices, self.chair_size)
        return output


def print_2d(matrix, chair_size):
    matrix = matrix.reshape(chair_size[0], chair_size[1])
    print("x", end=" | ")

    for i, _ in enumerate(matrix):
        print(i, end=" | ")

    print()
    for i, row in enumerate(matrix):
        print(i, end=" | ")
        for col in row:
            print(int(col), end=" | ")

        print()


def main():
    chair_size = (3, 3)
    model = ConflictModule(chair_size)

    gs, rgs = make_group(3, 3)
    ct = get_conflict_table(gs, 5)
    # print_2d(ct.reshape(1, -1), (9, 9))

    output = model.train_model(ct, rgs)
    # probabilities = softmax(output, dim=1)
    # persons = probabilities.argmax(dim=1)
    # print_2d(persons, chair_size)
    # a = model.forward(ct)
    # print(get_score(a, rgs, ct, (5, 5)))


def main2():
    chair_size = (5, 5)
    gs, rgs = make_group(5, 5)
    persons = torch.arange(25)
    print(rgs)
    for i, j in find_all_subgroups(persons, rgs, chair_size).items():
        print(i, j)


main()
