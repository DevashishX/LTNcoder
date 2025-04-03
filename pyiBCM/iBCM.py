print("in iBCM.py")


class Constraint:

    def __init__(self, name, a, b, w):
        self.name = name
        self.a = a
        self.b = b
        self.w = w

    def __str__(self):
        return self.name + "(" + str(self.a) + "-" + str(self.b) + ")" + str(self.w)

    def __hash__(self):
        atts = self.name + "," + str(self.a) + "," + str(self.b) + "," + str(self.w)
        return hash(atts)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Constraint):
            if (
                self.name == other.name
                and self.a == other.a
                and self.b == other.b
                and self.w == other.w
            ):
                return True
            else:
                return False


class Annotated_trace:

    def __init__(self, name, constraints, label):
        self.string = name
        self.constraints = constraints
        self.label = label


class ConstraintMining:

    def __init__(self, s, i, activities, w):
        self.text_string = s
        self.activities = activities
        self.local_constraints = set()
        self.i = i

        self.window_positions = []
        self.no_windows = w
        self.window_size = int(len(self.text_string) / w)

        if self.window_size > 0:
            for lw in range(1, w):
                self.window_positions.append(self.window_size * lw)
        self.window_positions.append(len(self.text_string))

    #        if i=='7':
    #            print('Window size: ', self.window_size)
    #            print('\n\n',s)

    def FindConstraints(self):
        act_positions = {}
        for a in self.activities:
            act_positions[a] = []

        for pos, act_a in enumerate(self.text_string):
            if act_a in act_positions.keys():
                act_positions[act_a].append(pos)
        #        print(act_positions)
        for window in range(0, self.no_windows):
            covered = set()
            for act_a in self.activities:
                #            if self.i=='7':
                #                print('\nAct: ', act_a)

                #                if self.i=='7':
                #                    print('Window ', window)
                lb = window * self.window_size
                ub = self.window_positions[window]

                a_list = self.get_window(act_positions[act_a], lb, ub)
                #                if self.i=='7':
                #                    print(a_list)
                if len(a_list) == 0:
                    #                    if self.i=='7':
                    #                        print('Adding absence')
                    self.local_constraints.add(
                        Constraint("absence", act_a, act_a, window)
                    )
                elif len(a_list) == 1:
                    self.local_constraints.add(
                        Constraint("exactly", act_a, act_a, window)
                    )
                elif len(a_list) == 2:
                    self.local_constraints.add(
                        Constraint("exactly2", act_a, act_a, window)
                    )
                else:
                    self.local_constraints.add(
                        Constraint("existence3", act_a, act_a, window)
                    )

                if len(a_list) > 0:
                    if a_list[0] == lb:
                        self.local_constraints.add(
                            Constraint("init", act_a, act_a, window)
                        )
                    if a_list[len(a_list) - 1] == ub:
                        self.local_constraints.add(
                            Constraint("last", act_a, act_a, window)
                        )
                    for act_b in self.activities:
                        covered.add((act_a, act_b))
                        if act_a != act_b and (act_b, act_a) not in covered:
                            #                            print('Mining for ', act_a,' and ', act_b)
                            b_list = self.get_window(act_positions[act_b], lb, ub)
                            if len(b_list) > 0:
                                self.local_constraints.add(
                                    Constraint("co_existence", act_a, act_b, window)
                                )
                                self.mine_binaries(act_a, act_b, a_list, b_list, window)
                                self.mine_binaries(act_b, act_a, b_list, a_list, window)
        return self.local_constraints

    def mine_binaries(self, act_a, act_b, a_list, b_list, win):
        p = ap = r = ar = cp = cr = False

        if b_list[0] > a_list[0]:
            if a_list[len(a_list) - 1] < b_list[0]:
                self.local_constraints.add(
                    Constraint("not_succession", act_b, act_a, win)
                )

            self.local_constraints.add(Constraint("precedence", act_a, act_b, win))
            p = True

            index = 0
            previous = next_i = b_list[0]
            go_on = True
            chain = (b_list[0] - 1) in a_list
            while go_on and (index + 1) < len(b_list):
                index += 1
                next_i = b_list[index]
                if (next_i - previous) > 1:
                    for i in range(previous + 1, next_i):
                        if i in a_list:
                            go_on = True
                            if (next_i - 1) not in a_list:
                                chain = False
                            break
                        go_on = False
                    previous = next_i
                else:
                    go_on = False
            if next_i == b_list[len(b_list) - 1] and go_on:
                if len(b_list) > 1:
                    self.local_constraints.add(
                        Constraint("alternate_precedence", act_a, act_b, win)
                    )
                    ap = True
                if chain:
                    self.local_constraints.add(
                        Constraint("chain_precedence", act_a, act_b, win)
                    )
                    cp = True

        if b_list[len(b_list) - 1] > a_list[len(a_list) - 1]:
            self.local_constraints.add(Constraint("response", act_a, act_b, win))
            r = True

            index = 0
            go_on = True
            previous = next_i = a_list[0]
            chain = (a_list[len(a_list) - 1] + 1) in b_list
            while go_on and (index + 1) < len(a_list):
                index += 1
                next_i = a_list[index]
                if (next_i - previous) > 1:
                    for i in range(previous + 1, next_i):
                        if i in b_list:
                            go_on = True
                            if (previous + 1) not in b_list:
                                chain = False
                            break
                        go_on = False
                    previous = next_i
                else:
                    go_on = False
            if next_i == a_list[len(a_list) - 1] and go_on:
                if len(a_list) > 1:
                    self.local_constraints.add(
                        Constraint("alternate_response", act_a, act_b, win)
                    )
                    ar = True
                if chain:
                    self.local_constraints.add(
                        Constraint("chain_response", act_a, act_b, win)
                    )
                    cr = True

        if p and r:
            self.local_constraints.add(Constraint("succession", act_a, act_b, win))
        if ap and ar:
            self.local_constraints.add(
                Constraint("alternate_succession", act_a, act_b, win)
            )
        if cp and cr:
            self.local_constraints.add(
                Constraint("chain_succession", act_a, act_b, win)
            )

    def get_window(self, a_list, lb, ub):
        output = []
        for i in range(lb, ub):
            if i in a_list:
                output.append(i)
        return output


def reduce_feature_space(constraints):
    print("Begin size: ", len(constraints))

    toRemove = set()

    lookAt = set()
    lookOut = set()
    for c in constraints:
        if "succession" in c.name:
            lookAt.add(c)
        if (
            "response" in c.name
            or "precedence" in c.name
            or "succession" in c.name
            or "co_existence" in c.name
        ):
            lookOut.add(c)

    for c in lookAt:
        for c2 in lookOut:
            if c.w == c2.w and c.a == c2.a and c.b == c2.b:
                if c.name == "succession" and (
                    c2.name == "response" or c2.name == "precedence"
                ):
                    toRemove.add(c2)
                if (
                    c.name == "alternate_succession"
                    and "chain" not in c.name
                    and (
                        "response" in c2.name
                        or "precedence" in c2.name
                        or c2.name == "succession"
                    )
                ):
                    toRemove.add(c2)
                if c.name == "chain_succession" and (
                    "response" in c2.name
                    or "precedence" in c2.name
                    or c2.name == "succession"
                    or c2.name == "alternate_succession"
                ):
                    toRemove.add(c2)
            if c.w == c2.w and (
                (c.a == c2.a and c.b == c2.b) or (c.b == c2.a and c.a == c2.b)
            ):
                if c.name == "succession" and c2.name == "co_existence":
                    toRemove.add(c2)
    print("Remove size (Succession removal): ", len(toRemove))
    constraints = constraints.difference(toRemove)
    toRemove = set()

    lookAt = set()
    for c in constraints:
        if "response" in c.name or "precedence" in c.name:
            lookAt.add(c)

    for c in lookAt:
        if c.name == "chain_response":
            for c2 in lookAt:
                if c.w == c2.w and c.a == c2.a and c.b == c2.b:
                    if c2.name == "response" or c2.name == "alternate_response":
                        toRemove.add(c2)
        if c.name == "chain_precedence":
            for c2 in lookAt:
                if c.w == c2.w and c.a == c2.a and c.b == c2.b:
                    if c2.name == "precedence" or c2.name == "alternate_precedence":
                        toRemove.add(c2)
        if c.name == "alternate_response":
            for c2 in lookAt:
                if c.w == c2.w and c.a == c2.a and c.b == c2.b:
                    if c2.name == "response":
                        toRemove.add(c2)
        if c.name == "alternate_precedence":
            for c2 in lookAt:
                if c.w == c2.w and c.a == c2.a and c.b == c2.b:
                    if c2.name == "precedence":
                        toRemove.add(c2)
    print("Remove size (Chain/Alternate removal): ", len(toRemove))
    constraints = constraints.difference(toRemove)
    toRemove = set()
    lookAt = set()
    for c in constraints:
        if "exactly" in c.name or "existence" in c.name:
            lookAt.add(c)

    for c in lookAt:
        if c.name == "existence3":
            for c2 in lookAt:
                if (
                    c.w == c2.w
                    and c.a == c2.a
                    and (c2.name == "existence2" or c2.name == "existence")
                ):
                    toRemove.add(c2)
        if c.name == "existence2":
            for c2 in lookAt:
                if c.w == c2.w and c.a == c2.a and c2.name == "existence":
                    toRemove.add(c2)
        if c.name == "exactly2":
            for c2 in lookAt:
                if (
                    c.w == c2.w
                    and c.a == c2.a
                    and (
                        c2.name == "existence"
                        or c2.name == "exactly"
                        or c2.name == "existence2"
                    )
                ):
                    toRemove.add(c2)
        if c.name == "exactly2":
            for c2 in lookAt:
                if (
                    c.w == c2.w
                    and c.a == c2.a
                    and (
                        c2.name == "existence"
                        or c2.name == "exactly"
                        or c2.name == "existence2"
                    )
                ):
                    toRemove.add(c2)
        if (
            "existence" in c.name or "exactly" in c.name
        ) and "co_existence" not in c.name:
            for c2 in lookAt:
                if (
                    c.w == c2.w
                    and c.a == c2.a
                    and ("existence" in c2.name or "exactly" in c2.name)
                    and "co_existence" not in c2.name
                ):
                    for c3 in constraints:
                        if (
                            c3.name == "co_existence"
                            and c2.w == c3.w
                            and (
                                (c.a == c3.a and c2.b == c3.b)
                                or (c.b == c3.a and c2.a == c3.b)
                            )
                        ):
                            toRemove.add(c3)
    print("Remove size (Unary removal): ", len(toRemove))
    constraints = constraints.difference(toRemove)
    print("End size: ", len(constraints))
    return constraints
