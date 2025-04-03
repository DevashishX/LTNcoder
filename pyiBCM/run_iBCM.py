print("in run_iBCM.py")
from iBCM import ConstraintMining, Annotated_trace, reduce_feature_space

VERBOSE = True


def load_dataset_mine_constraints(trace_file, label_file, min_sup, no_win):
    traces = []
    label_list = []
    labels = set()
    constraints_per_label = {}
    for trace, label in zip(trace_file, label_file):
        traces.append(trace)
        label_list.append(label.replace("\n", ""))
        labels.add(label.replace("\n", ""))

    traces_per_label = {}
    annotated_traces_per_label = {}
    for la, current_label in enumerate(sorted(labels)):
        if VERBOSE:
            print("\nCurrent label: ", current_label)
        final_traces = []
        activity_count = {}
        for label, trace in zip(label_list, traces):
            if label == current_label:
                trace = trace.replace("\n", "")
                trace = trace.replace(" -1 -2", "")

                #                print(trace)
                acts = trace.split(" -1 ")
                for act in acts:
                    if act not in activity_count.keys():
                        activity_count[act] = 0
                    activity_count[act] += 1
                final_traces.append(acts)
        traces_per_label[label] = final_traces

        non_redundant_activities = set()
        for act, count in activity_count.items():
            if count >= len(final_traces) * min_sup:
                non_redundant_activities.add(act)
        if VERBOSE:
            print("#non-redundant activities: ", len(non_redundant_activities))
        non_redundant_activities = sorted(non_redundant_activities)

        constraints_for_label, annotated_traces = mine_constraints(
            final_traces, non_redundant_activities, current_label, min_sup, no_win
        )
        annotated_traces_per_label[current_label] = annotated_traces

        constraints_per_label[current_label] = constraints_for_label
    return constraints_per_label, annotated_traces_per_label


def load_dataset_check_constraints(trace_file, label_file, activities, no_win):
    traces = []
    label_list = []
    labels = set()
    constraints_per_label = {}
    for trace, label in zip(trace_file, label_file):
        traces.append(trace)
        label_list.append(label.replace("\n", ""))
        labels.add(label.replace("\n", ""))

    traces_per_label = {}
    annotated_traces_per_label = {}
    for la, current_label in enumerate(labels):
        if VERBOSE:
            print("\nCurrent label: ", current_label)
        final_traces = []
        activity_count = {}
        for label, trace in zip(label_list, traces):
            if label == current_label:
                trace = trace.replace("\n", "")
                trace = trace.replace(" -1 -2", "")

                #                print(trace)
                acts = trace.split(" -1 ")
                for act in acts:
                    if act not in activity_count.keys():
                        activity_count[act] = 0
                    activity_count[act] += 1
                final_traces.append(acts)
        traces_per_label[label] = final_traces

        constraints_for_label, annotated_traces = mine_constraints(
            final_traces, activities, current_label, 0, no_win
        )

        constraints_per_label[current_label] = constraints_for_label
        annotated_traces_per_label[current_label] = annotated_traces

    return constraints_per_label, annotated_traces_per_label


def mine_constraints(traces, non_redundant_activities, label, min_sup, no_win):
    if VERBOSE:
        print("#traces", len(traces))
    too_short = 0
    constraint_count = {}
    actual_traces = 0
    annotated_traces = []

    # print('Non redundant:', non_redundant_activities)

    for t, trace in enumerate(traces):
        if t % 10000 == 0 and t > 0:
            print("Doing trace", t)
        constraints = set()
        if len(trace) >= no_win:
            actual_traces += 1
            miner = ConstraintMining(trace, label, non_redundant_activities, no_win)
            constraints = miner.FindConstraints()

            # print('Trace:', trace)
            # for con in constraints:
            #     print(con)

            for constraint in constraints:
                if constraint not in constraint_count.keys():
                    constraint_count[constraint] = 0
                constraint_count[constraint] += 1
            # annotated_traces[t] = constraints
            annotated_traces.append(Annotated_trace(trace, constraints, label))
        else:
            too_short += 1
    if VERBOSE:
        print("Too short: ", too_short)
        print("#constraints prior removal: ", len(constraint_count))

    to_remove = set()
    for constraint, count in constraint_count.items():
        # print('Actual traces:', actual_traces)
        if count < (actual_traces * min_sup):
            to_remove.add(constraint)
    for tr in to_remove:
        del constraint_count[tr]

    return constraint_count.keys(), annotated_traces


def iBCM(filename, traces, labels, rfs, min_sup=0.01, no_win=1):
    print("\nRunning iBCM training")

    constraints = set()

    constraints_per_label, annotated_traces = load_dataset_mine_constraints(
        traces, labels, min_sup, no_win
    )

    print("\n**********************\nFinal stats: ")

    all_constraints = set()
    for label, constraints in constraints_per_label.items():
        print("Label ", label, " has ", len(constraints), " constraints")
        all_constraints = all_constraints.union(set(constraints))
    print("\nTotal #constraints: ", len(all_constraints))

    joint_features = set()
    for constraint in all_constraints:
        contained = True
        for label, constraints in constraints_per_label.items():
            if constraint not in constraints:
                contained = False
                break
        if contained:
            joint_features.add(constraint)
    print("Joint features:", len(joint_features))
    all_constraints = all_constraints.difference(joint_features)

    if rfs:
        final_constraints = reduce_feature_space(all_constraints)
    else:
        final_constraints = all_constraints
    print("Final total #constraints: ", len(final_constraints))

    return final_constraints


def iBCM_verify(filename, traces, labels, constraints, no_win=1):
    print(
        "\nRunning iBCM labelling ",
        filename,
        " for (",
        len(constraints),
        " constraints)",
    )

    used_activities = set()
    for constraint in constraints:
        used_activities.add(constraint.a)
        used_activities.add(constraint.b)

    # print('Used activities:', len(used_activities))
    # print('Used constraints:', len(constraints))

    constraints_per_label, annotated_traces = load_dataset_check_constraints(
        traces, labels, used_activities, no_win
    )

    for label in annotated_traces.keys():
        for trace in annotated_traces[label]:
            trace = trace.constraints.intersection(constraints)

    output = open(filename, "w")
    for constraint in constraints:
        output.write(str(constraint) + ",")
    output.write("label\n")

    for label in annotated_traces.keys():
        for trace in annotated_traces[label]:
            for constraint in constraints:
                if constraint in trace.constraints:
                    output.write("1,")
                else:
                    output.write("0,")
            output.write(label + "\n")
    output.close()
