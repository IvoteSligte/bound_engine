# Bound Engine
An incomplete 3D light simulation engine.

I ended up experimenting with various novel light simulation methods, including conversion of scenes to sphere-based binary bounding volume hierarchies rendered with path-tracing, a couple sphere marching implementations with several unique optimizations, rendering combined with spherical harmonic world-space caching, and particle-based solutions.

All of these were "big idea" methods, rather than a combination of specialized methods, resulting in any combination of the following flaws: poor scalability, slow convergence, unrealistic results, and obvious Moiré patterns.

They did teach me a lot about light transport methods, though, and inspired me to study systems such as Unreal Engine's Lumen, which for example inspired me to implement LODs for world-space caches in several of my "big idea" attempts.
