import ltn
import ltn.core
import tensorflow as tf

eps = 1e-4
Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
MultiAnd = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_ProdMulti())
MultiOr = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSumMulti())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=4),semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=5),semantics="exists")
formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))

class AxiomBuilder:
    """
    A class to build axioms for anomaly detection using LTN.
    """

    def __init__(self, activity_predicates: list[ltn.Predicate], activity_constants:dict[str, ltn.Constant]):
        """Save the set of activity predicates and constants.
        This is used to build axioms for anomaly detection.

        Args:
            traces (ltn.Variable): traces variable which contains traces
            activity_predicates (list[ltn.Predicate]): list of activity predicates_
            activity_constants (dict[str, ltn.Constant]): dictionary of activity constants_
        """
        self.activity_predicates = activity_predicates
        self.activity_constants = activity_constants
        self.total_activity_predicates = len(activity_predicates)
        self.axioms = []
        pass

    def build_axiom_fn(self):
        
        @tf.function
        def axioms(features, labels=None, training=False):
            traces = ltn.Variable("traces", features)
            axioms = self._build_axioms(traces, training=training)
            sat_level = formula_aggregator(axioms).tensor
            return sat_level
        return axioms
    
    def _build_axioms(self, traces:ltn.Variable, training):
        """
        Build an axiom for anomaly detection.

        """
        axioms = []
        
        # Responded Existence[Develop Method, Final Decision] | |
        # self.axioms.append(self.responded_existence("Develop Method", "Final Decision", traces, training))
        # Response[Develop Method, Final Decision] | |
        axioms.append(self.response("Develop Method", "Final Decision", traces, training))
        # Chain Precedence[Research Related Work, Develop Method] | |
        # self.axioms.append(self.chain_precedence("Research Related Work", "Develop Method", traces, training))
        # Chain Response[Develop Method, Experiment] | |
        # self.axioms.append(self.chain_response("Develop Method", "Experiment", traces, training))

        return axioms
    
    def _existence(self, activity_constant_key:str, traces:ltn.Variable, training) -> ltn.core.Formula:
        formula = MultiOr(*[self.activity_predicates[i]([traces, self.activity_constants[activity_constant_key]], training=training) for i in range(self.total_activity_predicates)])
        return formula
    
    def existence(self, activity_constant_key:str, traces:ltn.Variable, training) -> ltn.core.Formula:
        formula = Forall(traces, self._existence(activity_constant_key, traces, training))
        print(f"Built forall existence({activity_constant_key})")
        return formula
    
    def _absence(self, activity_constant_key:str, traces:ltn.Variable, training) -> ltn.core.Formula:
        # formula = Not(MultiOr(*[self.activity_predicates[i]([traces, self.activity_constants[activity_constant_key]], training=training) for i in range(self.total_activity_predicates)]))
        formula = Not(self._existence(activity_constant_key, traces, training))
        return formula
    
    def absence(self, activity_constant_key:str, traces:ltn.Variable, training) -> ltn.core.Formula:
        formula = Forall(traces, self._absence(activity_constant_key, traces, training))
        print(f"Built forall absence({activity_constant_key})")
        return formula
    
    def _choice(self, activity_constant_key_a:str, activity_constant_key_b:str, traces:ltn.Variable, training) -> ltn.core.Formula:
        formula = Or(
            self._existence(activity_constant_key_a, traces, training),
            self._existence(activity_constant_key_b, traces, training)
        )
        return formula
    
    def choice(self, activity_constant_key_a:str, activity_constant_key_b:str, traces:ltn.Variable, training) -> ltn.core.Formula:
        formula = Forall(traces, self._choice(activity_constant_key_a, activity_constant_key_b, traces, training))
        print(f"Built forall choice({activity_constant_key_a}, {activity_constant_key_b})")
        return formula
    
    def _exclusive_choice(self, activity_constant_key_a:str, activity_constant_key_b:str, traces:ltn.Variable, training) -> ltn.core.Formula:
        formula = Or(
            And(
                self._existence(activity_constant_key_a, traces, training),
                self._absence(activity_constant_key_b, traces, training)
            ),
            And(
                self._absence(activity_constant_key_a, traces, training),
                self._existence(activity_constant_key_b, traces, training)
            )
        )
        return formula
    
    def exclusive_choice(self, activity_constant_key_a:str, activity_constant_key_b:str, traces:ltn.Variable, training) -> ltn.core.Formula:
        formula = Forall(traces, self._exclusive_choice(activity_constant_key_a, activity_constant_key_b, traces, training))
        print(f"Built forall exclusive_choice({activity_constant_key_a}, {activity_constant_key_b})")
        return formula
    
    def _responded_existence(self, activity_constant_key_a: str, activity_constant_key_b: str, traces: ltn.Variable, training) -> ltn.core.Formula:
    
        exists_a = self._existence(activity_constant_key_a, traces, training)
        exists_b = self._existence(activity_constant_key_b, traces, training)
        implication = Implies(exists_a, exists_b)
        return implication
    
    def responded_existence(self, activity_constant_key_a: str, activity_constant_key_b: str, traces: ltn.Variable, training) -> ltn.core.Formula:
        formula = Forall(traces, self._responded_existence(activity_constant_key_a, activity_constant_key_b, traces, training))
        print(f"Built forall responded_existence({activity_constant_key_a}, {activity_constant_key_b})")
        return formula
    
    # def _responded_existence(self, activity_constant_key_a:str, activity_constant_key_b:str, traces:ltn.Variable, training) -> ltn.core.Formula:
    #     """
    #     Forall i in [0, len(activity_predicates)](
    #         activity_predicates[i](traces, activity_constants[activity_constant_key_a], training=training) 
    #         => Exists j in [0, len(activity_predicates)] (
    #             activity_predicates[j](traces, activity_constants[activity_constant_key_b], training=training))
    #     )
    #     """
    #     formula = None # TODO: Implement this function using above description
    #     return formula
    
    def _response(self, activity_constant_key_a: str, activity_constant_key_b: str, traces: ltn.Variable, training) -> ltn.core.Formula:
        """
        If activity A occurs at position i, then activity B must occur at some position j >= i.
        
        ∀i: pred_i(trace, A) ⇒ ∃j≥i: pred_j(trace, B)
        """
        formulas = []

        for i in range(self.total_activity_predicates):
            # Predicate for activity A at position i
            pred_a_i = self.activity_predicates[i]([traces, self.activity_constants[activity_constant_key_a]], training=training)

            # All B predicates from position i onward
            future_preds_b = [
                self.activity_predicates[j]([traces, self.activity_constants[activity_constant_key_b]], training=training)
                for j in range(i, self.total_activity_predicates)
            ]

            # ∃j ≥ i: pred_j(trace, B)
            exists_b = MultiOr(*future_preds_b)

            # pred_i(trace, A) ⇒ ∃j ≥ i: pred_j(trace, B)
            implication = Implies(pred_a_i, exists_b)

            formulas.append(implication)

        # ∀trace: all implications across positions
        return MultiAnd(*formulas)
    
    def response(self, activity_constant_key_a: str, activity_constant_key_b: str, traces: ltn.Variable, training) -> ltn.core.Formula:
        formula = Forall(traces, self._response(activity_constant_key_a, activity_constant_key_b, traces, training))
        print(f"Built forall response({activity_constant_key_a}, {activity_constant_key_b})")
        return formula
        

    def _alternate_response(self, activity_constant_key_a: str, activity_constant_key_b: str, traces: ltn.Variable, training) -> ltn.core.Formula:
        """
        For every i: if A at i, then ∃j > i where B occurs and no A occurs in between.
        """
        # Step 1: Cache all predicate outputs across positions for A and B
        a_preds = [
            self.activity_predicates[i]([traces, self.activity_constants[activity_constant_key_a]], training=training)
            for i in range(self.total_activity_predicates)
        ]
        b_preds = [
            self.activity_predicates[i]([traces, self.activity_constants[activity_constant_key_b]], training=training)
            for i in range(self.total_activity_predicates)
        ]

        implications = []

        for i in range(self.total_activity_predicates):
            a_i = a_preds[i]

            # All valid j > i such that:
            #    - B occurs at j
            #    - ∀k ∈ (i, j): ¬A_k
            valid_paths = []

            for j in range(i + 1, self.total_activity_predicates):
                b_j = b_preds[j]

                # All ¬A_k for i < k < j
                if j > i + 1:
                    not_a_k = [Not(a_preds[k]) for k in range(i + 1, j)]
                    valid = And(*not_a_k, b_j)
                else:
                    valid = b_j  # no steps in between, directly B

                valid_paths.append(valid)

            # If no valid j > i, return unsatisfied formula
            if valid_paths:
                b_after_a = MultiOr(*valid_paths)
            else:
                b_after_a = ltn.Proposition(0.0 + eps, trainable=False)  # unsatisfied if no valid j

            implication = Implies(a_i, b_after_a)
            implications.append(implication)

        return MultiAnd(*implications)
    
    def alternate_response(self, activity_constant_key_a: str, activity_constant_key_b: str, traces: ltn.Variable, training) -> ltn.core.Formula:
        formula = Forall(traces, self._alternate_response(activity_constant_key_a, activity_constant_key_b, traces, training))
        print(f"Built forall alternate_response({activity_constant_key_a}, {activity_constant_key_b})")
        return formula


    def _chain_response(self, activity_constant_key_a: str, activity_constant_key_b: str, traces: ltn.Variable, training) -> ltn.core.Formula:
        """
        For every i in [0, n-2]: A_i ⇒ B_{i+1}
        """
        # Step 1: Cache all predicate evaluations
        a_preds = [
            self.activity_predicates[i]([traces, self.activity_constants[activity_constant_key_a]], training=training)
            for i in range(self.total_activity_predicates)
        ]
        b_preds = [
            self.activity_predicates[i]([traces, self.activity_constants[activity_constant_key_b]], training=training)
            for i in range(self.total_activity_predicates)
        ]

        implications = []

        # Only go up to n-2 because we check i and i+1
        for i in range(self.total_activity_predicates - 1):
            a_i = a_preds[i]
            b_next = b_preds[i + 1]
            implication = Implies(a_i, b_next)
            implications.append(implication)

        return MultiAnd(*implications)
    
    def chain_response(self, activity_constant_key_a: str, activity_constant_key_b: str, traces: ltn.Variable, training) -> ltn.core.Formula:
        formula =  Forall(traces, self._chain_response(activity_constant_key_a, activity_constant_key_b, traces, training))
        print(f"Built forall chain_response({activity_constant_key_a}, {activity_constant_key_b})")
        return formula

    def _chain_precedence(self, activity_constant_key_a: str, activity_constant_key_b: str, traces: ltn.Variable, training) -> ltn.core.Formula:
        """
        For every i in [1, n-1]: B_i ⇒ A_{i-1}
        """
        # Step 1: Cache all predicate evaluations
        a_preds = [
            self.activity_predicates[i]([traces, self.activity_constants[activity_constant_key_a]], training=training)
            for i in range(self.total_activity_predicates)
        ]
        b_preds = [
            self.activity_predicates[i]([traces, self.activity_constants[activity_constant_key_b]], training=training)
            for i in range(self.total_activity_predicates)
        ]

        implications = []

        # Start from i = 1 because we check i and i-1
        for i in range(1, self.total_activity_predicates):
            b_i = b_preds[i]
            a_prev = a_preds[i - 1]
            implication = Implies(b_i, a_prev)
            implications.append(implication)

        return MultiAnd(*implications)
    
    def chain_precedence(self, activity_constant_key_a: str, activity_constant_key_b: str, traces: ltn.Variable, training) -> ltn.core.Formula:
        formula = Forall(traces, self._chain_precedence(activity_constant_key_a, activity_constant_key_b, traces, training))
        print(f"Built forall chain_precedence({activity_constant_key_a}, {activity_constant_key_b})")
        return formula

    def _precedence(self, activity_constant_key_a: str, activity_constant_key_b: str, traces: ltn.Variable, training) -> ltn.core.Formula:
        """
        For every j: if B occurs at j, then ∃i < j where A occurred.
        """
        a_preds = [
            self.activity_predicates[i]([traces, self.activity_constants[activity_constant_key_a]], training=training)
            for i in range(self.total_activity_predicates)
        ]
        b_preds = [
            self.activity_predicates[i]([traces, self.activity_constants[activity_constant_key_b]], training=training)
            for i in range(self.total_activity_predicates)
        ]

        implications = []

        for j in range(self.total_activity_predicates):
            b_j = b_preds[j]

            # All i < j where A(i) is true
            valid_precedents = [a_preds[i] for i in range(j)]
            if valid_precedents:
                a_before_b = MultiOr(*valid_precedents)
            else:
                # a_before_b = ltn.Constant(tf.constant(0.0), trainable=False)  # unsatisfied if j == 0
                a_before_b = ltn.Proposition(0.0 + eps, trainable=False)

            implication = Implies(b_j, a_before_b)
            implications.append(implication)

        return MultiAnd(*implications)    
    
    def precedence(self, activity_constant_key_a: str, activity_constant_key_b: str, traces: ltn.Variable, training) -> ltn.core.Formula:
        formula = Forall(traces, self._precedence(activity_constant_key_a, activity_constant_key_b, traces, training))     
        print(f"Built forall precedence({activity_constant_key_a}, {activity_constant_key_b})")
        return formula
    
    def _not_response(self, activity_constant_key_a: str, activity_constant_key_b: str, traces: ltn.Variable, training) -> ltn.core.Formula:
        """
        For all i: if A happens at i, then B must not happen at any j > i.
        """
        a_preds = [
            self.activity_predicates[i]([traces, self.activity_constants[activity_constant_key_a]], training=training)
            for i in range(self.total_activity_predicates)
        ]
        b_preds = [
            self.activity_predicates[i]([traces, self.activity_constants[activity_constant_key_b]], training=training)
            for i in range(self.total_activity_predicates)
        ]

        constraints = []

        for i in range(self.total_activity_predicates):
            a_i = a_preds[i]

            # All B(j) for j > i
            if i < self.total_activity_predicates - 1:
                future_b = [b_preds[j] for j in range(i + 1, self.total_activity_predicates)]
                no_b_after = MultiAnd(*[Not(b) for b in future_b])
            else:
                # no_b_after = ltn.Constant(tf.constant(1.0), trainable=False)  # vacuously true if no future
                no_b_after = ltn.Proposition(1.0 - eps, trainable=False)  # vacuously true if no future
                

            implication = Implies(a_i, no_b_after)
            constraints.append(implication)

        return MultiAnd(*constraints)
    
    def not_response(self, activity_constant_key_a: str, activity_constant_key_b: str, traces: ltn.Variable, training) -> ltn.core.Formula:
        formula = Forall(traces, self._not_response(activity_constant_key_a, activity_constant_key_b, traces, training))
        print(f"Built forall not_response({activity_constant_key_a}, {activity_constant_key_b})")
        return formula
    
    def _not_precedence(self, activity_constant_key_a: str, activity_constant_key_b: str, traces: ltn.Variable, training) -> ltn.core.Formula:
        """
        For every j: if B happens at j, then A must NOT happen at any i < j.
        """
        a_preds = [
            self.activity_predicates[i]([traces, self.activity_constants[activity_constant_key_a]], training=training)
            for i in range(self.total_activity_predicates)
        ]
        b_preds = [
            self.activity_predicates[i]([traces, self.activity_constants[activity_constant_key_b]], training=training)
            for i in range(self.total_activity_predicates)
        ]

        constraints = []

        for j in range(self.total_activity_predicates):
            b_j = b_preds[j]

            # All A(i) for i < j → we want NOT(A(i)) for each
            if j > 0:
                no_a_before = MultiAnd(*[Not(a_preds[i]) for i in range(j)])
            else:
                no_a_before = ltn.Proposition(1.0 - eps, trainable=False)  # vacuously true for j = 0

            implication = Implies(b_j, no_a_before)
            constraints.append(implication)

        return MultiAnd(*constraints)

    def not_precedence(self, activity_constant_key_a: str, activity_constant_key_b: str, traces: ltn.Variable, training) -> ltn.core.Formula:
        formula = Forall(traces, self._not_precedence(activity_constant_key_a, activity_constant_key_b, traces, training))
        print(f"Built forall not_precedence({activity_constant_key_a}, {activity_constant_key_b})")
        return formula
    
    
    def _build_test_axiom(self, activity_constant_key:str, traces:ltn.Variable, training):
        axioms = []
        axioms.append(
                Forall(traces, self.activity_predicates[0]([traces, self.activity_constants["▶"]], training=training))
        )
        for constant_key, constant in self.activity_constants.items():
                if constant_key != activity_constant_key:
                    axioms.append(
                        Forall(traces, 
                               Not(
                                   self.activity_predicates[0](
                                       [traces, constant], training=training
                                       )
                                   )
                               )
                        )
        print(f"Built test axiom for {activity_constant_key}")
        return axioms
