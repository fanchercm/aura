These are the questions that candidate designs should survive.

For each proposed architecture, ask:

* Can this represent shared structure without manually wiring tens of thousands of links?
* Can it avoid parameter explosion?
* Can it recover known truth on synthetic data?
* Can it exploit angular structure rather than just store it?
* Can it process a reduced realistic dataset in practical time?
* Can a scientist understand why it failed?
* Can it degrade gracefully to a conventional 1D workflow for benchmarking?

If a design fails two or three of those early, it is probably not worth prototyping.