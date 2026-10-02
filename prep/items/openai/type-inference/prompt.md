A *type* is either atomic or a tuple. An atomic type is named by a string: one of the five primitives `int`, `float`, `str`, `bool`, `char`, or a generic such as `T1`, `T2`, `T3`, .... A tuple type holds zero or more types, nested to any depth. A *function* has a list of parameter types and one return type:

```text
Type      ::= Primitive | Generic | Tuple
Primitive ::= "int" | "float" | "str" | "bool" | "char"
Generic   ::= "T" Digit+
Tuple     ::= "[" [Type ("," Type)*] "]"
Function  ::= "(" [Type ("," Type)*] ")" "->" Type
```

A type is given directly as a Python object, never as text to parse:

```py
class Node:
    def __init__(self, value: Union[str, List["Node"]]) -> None:
        """An atomic node: value is its name. A tuple node: value is the list of its element
        types, possibly empty."""

class Function:
    def __init__(self, parameters: List[Node], return_type: Node) -> None: ...
```

### Part 1 — String representation

Implement `Node.__str__`, `Function.__str__`, and structural equality `Node.__eq__` (two nodes are equal exactly when their string forms are equal). An atomic node's string is its name. A tuple node's string lists its elements separated by commas with no space after a comma, inside square brackets: `[t1,t2,...]`, and `[]` for an empty tuple. A function's string lists its parameters the same way inside parentheses, then one space, `->`, one space, then the return type: `(p1,p2,...) -> returnType`.

```py
def __str__(self) -> str: ...        # Node
def __eq__(self, other) -> bool: ... # Node
def __str__(self) -> str: ...        # Function
```

Example: the function taking the tuple `[T1, bool]`, then `T2`, then `char`, and returning the tuple `[T2, [T1, int]]` prints as:

```text
([T1,bool],T2,char) -> [T2,[T1,int]]
```

### Part 2 — Return-type inference

The five primitives above are fixed, and every other atomic name is a *generic*: what counts is whether a name is a primitive, not the spelling `T` followed by digits. A call supplies one concrete type per declared parameter, in a list `parameters`; a concrete type never contains a generic. Implement:

```py
def get_return_type(parameters: List[Node], function: Function) -> Node: ...
```

It matches each of `function.parameters[i]` against `parameters[i]` (*unification*): a generic in the declared type is bound to whatever concrete type occupies the same position in the call, every occurrence of the same generic name within one call must bind to the same (structurally equal) type, and a generic may bind to an entire tuple, not only to a primitive. It then returns `function.return_type` with every generic replaced by its binding (*substitution*). Three errors are possible:

- `ArityError` — `len(parameters) != len(function.parameters)`.
- `TypeMismatchError` — at some position with no generic involved, the two types differ: a different primitive, or tuples of different length.
- `GenericConflictError` — the same generic name would have to bind to two structurally different types within one call.

Examples, using the Part 1 string form as shorthand (`call` lists the concrete argument types in order):

```text
function: (T1,bool,T2) -> [T2,T1]
call: char, bool, float
-> [float,char]

function: (T1,[T1,bool],T2) -> [T2,[T1,T1]]
call: char, [char,bool], int
-> [int,[char,char]]

function: (T1,bool) -> [T1,T1]
call: [int,float], bool
-> [[int,float],[int,float]]

function: (T1,int,bool) -> T1
call: char, int
-> ArityError: expected 3 arguments, got 2

function: ([T1,int],bool) -> T1
call: [char,int,bool], bool
-> TypeMismatchError: expected [T1,int] but received [char,int,bool]

function: (int,T3,T3) -> T3
call: int, bool, char
-> GenericConflictError: T3 is bound to two different types: bool and char
```
