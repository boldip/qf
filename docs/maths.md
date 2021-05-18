## Contents

- [Definitions](#def)
- [Basic properties of fibrations](#prop)


## <a name=def>Definitions</a>

### Graphs

A _graph_ is a defined by _G=(N<sub>G</sub>, A<sub>G</sub>, s<sub>G</sub>, t<sub>G</sub>)_ (subscripts are omitted whenever they are clear from the context) where _N_ is a set of _nodes_, _A_ is a set of _arcs_, _s: A &rarr; N_ is a function assigning a _source_ node to every arc, and _t: A &rarr; N_ is a function assigning a _target_ node to every arc. 

This is pretty much like a normal directed graph (or digraph, or network, as you like) with the only difference that you may have many arcs between the same source/target pair. Graphs of this kind are sometimes called _multigraphs_, but since in our context this is what we need we prefer to stick to the simpler term _graph_.

### Graph morphism

Given two graphs _G_ and _H_, a _(graph) morphism_ 
_&phi;: G &rarr; H_ is a pair of maps _&phi;<sub>N</sub>: N<sub>G</sub> &rarr; N<sub>H</sub>_ and
&phi;<sub>A</sub>: A<sub>G</sub> &rarr; A<sub>H</sub>_ (again, the subscripts will be omitted when they are clear), such that _t(&phi;(a))=&phi;(t(a))_
and _s(&phi;(a))=&phi;(s(a))_ for every arc _a_ of _G_.

We say that the morphism is an _epimorphism_ if both maps are surjective, and an _isomorphism_ if both maps are bijective. An isomorphism from a graph to itself is called an _automorphism_. Automorphisms form a group under function composition.

### Morphisms and fibres

Let _&xi;: G &rarr; B_ be a graph morphism. For every node _x_ of _B_, the set of nodes of _G_ that are mapped to _x_ are called the _fibre of &xi; over x_. The fibres of &xi; form a partition of the set of nodes (i.e., an equivalence relation between the nodes) of _G_.

### Excess and deficiency

Let _&xi;: G &rarr; B_ be a graph morphism, _a_ be an arc of _B_ and _y_ be a node of _G_ such that _&xi;(y)=t(a)_ (that is, _y_ is in the fibre of the target of _a_). Consider the set _X(a,x)_ of arcs of _G_ that have target _y_ and that are mapped by _&xi;_ to _a_. We say that this pair _(a,x)_ has a 

- _deficiency_ of 1 if the set _X(a,x)_ is empty
- _excess_ of _k_ if the set _X(a,x)_ has cardinality _k+1&ge;2_.

The sum of all deficiencies/excesses is called the global deficiency/excess of the morphism _&xi;_. The sum of the global deficiency and excess of _&xi;_ is called the _total error_ of _&xi;_.

### Fibrations and quasi-fibrations

A _fibration_ is a morphism with total error 0. Formally (and equivalently) it can be defined as a morphism such that for every arc _a_ of the base and every node _y_ in the fibre of its target, there is precisely one arc of the upper graph that has target _y_ and image _a_.
Any morphism with small total error may be called _quasi-fibration_. Since there is no bound on how large the total error can be, in fact the word quasi-fibration can be used as a synonym of morphism, but when we say "quasi-fibration" we somehow imply that we want its total error to be small.

## <a name=prop>Basic properties of fibrations</a>

### Local in-isomorphisms

Given a graph _G_, an equivalence relation _&cong;_ between its nodes is called a _local in-isomorphism_ iff for any two nodes _x_,_x'_ such that _x&cong;x'_
there is a bijection between the arcs _a_ with target _x_ and the arcs _a'_ with target _x'_ such that _s(a)&cong;s(a')_.

The following holds:

**THEOREM**:  _&cong;_ is a local in-isomorphism for the nodes of _G_ iff there is a graph _B_ and a fibration _&xi;: G &rarr; B_ whose fibres are _&cong;_.

So, local in-isomorphism is a concept that is strictly related to that of fibration, although, a fibration is richer in the sense that it also describes an equivalence relation between arcs (something that the local in-isomorphism concept does not include).

### Minimum base 

We think of a fibration _&phi;: G &rarr; B_ as a way to "squeeze" _G_. It is natural to think if there is a somehow maximal way to squeeze _G_, in the precise sense that its associated equivalence relation is the coarsest possible (i.e., one that puts together as many pairs of nodes as possible).
The answer is positive:

**THEOREM**:  Given a graph _G_ there exists a graph _B_ (called the _minimum base of G_) and an epimorphic fibration _&mu;: G &rarr; B_ such that:

- every epimorphic fibration from _B_ is an isomorphism (i.e., _B_ cannot be nontrivially fibred);
- any other _&mu;': G &rarr; B'_ with the same property induces the same local in-isomorphism as _&mu;_ on the nodes of _G_.


What the above theorem says, essentially, is that there is a unique coarsest local in-isomorphism. And, a unique (up-to isomorphism) minimum epimorphic-fibration base. The fibration _&mu;_ is not unique, though, albeit only the arc-component can change.


### Constructing the minimum base

The last theorem can be turned into a construction, that is, into an algorithm that in fact builds the minimum base for a given graph. 
The simplest (although not particularly effective) way to build the minimum base uses the concept of _universal total graph_ (a.k.a., _view_): given a graph _G_ and a node _x_, consider the following rooted in-tree _VIEW(G,x)_:

- its nodes are the paths in _G_ ending at _x_ (by this we mean "all the paths", including the non-simple ones); a path is a finite sequence of arcs such that the target of every arc is the source of the following arc; this set of paths is typically infinite;
- a path _(a<sub>1</sub>,a<sub>2</sub>,...,a<sub>k</sub>)_ is a child of _(a<sub>2</sub>,...,a<sub>k</sub>)_ (the root being the empty path, which has no parent).
 
Define now an equivalence relation _&equiv;_ on the nodes of _G_ where two nodes _x_ and _y_ are equivalent iff they have isomorphic views (i.e., if the two trees _VIEW(G,x)_ and _VIEW(G,y)_ are isomorphic trees).

The minimum base _B_ can be built explicitly as follows:

- its nodes are the equivalence classes of _&equiv;_
- between class _X_ and class _Y_ there are as many arcs as there are arcs from nodes of _X_ to any specific node of _Y_ (by definition, this number is independent on the way you choose that specific node in _Y_).

As we said, this algorithm is not especially efficient; you may even wonder that it is in fact an algorithm because it requires to decide if two (potentially infinite) trees are isomorphic. It can be proven, though, that two views are isomorphic if and only if they are isomorphic when truncated after _n_ levels (where _n_ is the number of nodes of _G_), which makes the decision algorithmically sound.

There are other and more efficient ways to accomplish the same job; the best known is Cardon-Crochemore's algorithm [Cardon, Alain, and Maxime Crochemore. *Partitioning a graph in O (¦ A¦ log2¦ V¦)*. Theoretical Computer Science 19.1 (1982): 85-98]. We shall often use _Cardon-Crochemore_ to refer to the relation _&equiv;_ defined above, albeit, in principle, there is no need to use Cardon-Crochemore's algorithm to obtain it.





