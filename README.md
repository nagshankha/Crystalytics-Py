# Crystallography package

This package takes two fundamental inputs describing any crystal structure, viz. primitive vectors and structural motif. By structural motif, I mean the difference in atom types are not considered yet, but might be added later. 

Given a certain direction, the methods of class *CrystalStructure* calculates the lattice spacing and multiplicity of lattice site in that direction. Method *relativeDisplacementsLatticePlanes()* calculates the relative shifts in the lattice planes perpendicular to the given direction.




