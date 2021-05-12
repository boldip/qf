# qf

A package related to building quasi-fibration symmetries.

If you'd like to learn more about how it works, see the brief explanation and References below.

Brought to you by [Paolo Boldi](http://boldi.di.unimi.it/) (<paolo.boldi@unimi.it>).

Read the [full documentation](docs/build/html/index.html) for more information.


## Definitions

### Graphs

A _graph_ is a defined by _G=(N<sub>G</sub>, A<sub>G</sub>, s<sub>G</sub>, t<sub>G</sub>)_ (subscripts are omitted whenever they are clear from the context) where _N_ is a set of _nodes_, _A_ is a set of _arcs_, _s: A &rarr; N_ is a function assigning a _source_ node for every arc, and _t: A &rarr; N_ is a function assigning a _target_ node for every arc. 

This is pretty much like a normal directed graph (or digraph, or network, as you like) with the only difference that you may have many arcs between the same source/target pair.

### Graph morphism

Given two graphs _G_ and _H_, a _(graph) morphism_ 
_&phi;: G &rarr; H_ is a pair of maps _&phi;<sub>N</sub>: N<sub>G</sub> &rarr; N<sub>H</sub>_ and
&phi;<sub>A</sub>: A<sub>G</sub> &rarr; A<sub>H</sub>_ (again, the subscripts will be omitted when they are clear), such that _t(&phi;(a))=&phi;(t(a))_
and _s(&phi;(a))=&phi;(s(a))_ for every arc _a_ of _G_.

We say that the morphism is an _epimorphism_ if both maps are surjective, and an _isomorphism_ if both maps are bijective. An isomorphism from a graph to itself is called an _automorphism_. Automorphisms form a group under function composition.

### Morphisms and fibres

Let _&xi;: G &rarr; B_ be a graph morphism. For every node _x_ of _B_, the set of nodes of _G_ that are mapped to _x_ are called the _fibre of &xi; over x_. The fibres of &xi; are a partition of the set of nodes (i.e., an equivalence relation between the nodes) of _G_.

### Excess and deficiency

Let _&xi;: G &rarr; B_ be a graph morphism, _a_ be an arc of _B_ and _y_ be a node of _G_ such that _&xi;(y)=t(a)_. Consider the set of arcs of _G_ that have target _y_ and that are mapped by _&xi;_ to _a_. We say that this pair _(a,x)_ has a 

- _deficiency_ of 1 if the set is empty
- _excess_ of _k_ if the set has cardinality _k+1&ge;2_.

The sum of all deficiencies/excesses is called the global deficiency/excess of the morphism &xi;. The sum of deficiency and excess is called the _total error_ of &xi;.

### Fibrations and quasi-fibrations

A _fibration_ is a morphism with total error 0.
Any morphism with small total error may be called _quasi-fibration_. Since there is no bound on how large the total error can be, in fact the word quasi-fibration can be used as a synonym of morphism, but when we say "quasi-fibration" we somehow imply that we want its total error to be small.

## Properties

### Local in-isomorphisms

Given a graph _G_, an equivalence relation _&cong;_ between its nodes is called a _local in-isomorphism_ iff for any two nodes _x_,_x'_ such that _x&cong;x'_
there is a bijection between the arcs _a_ with target _x_ and the arcs _a'_ with target _x'_ such that _s(a)&cong;s(a')_.

The following holds:

**THEOREM**:  _&cong;_ is a local in-isomorphism for the nodes of _G_ iff there is a graph _B_ and a fibration _&xi;: G &rarr; B_ whose fibres are _&cong;_.

