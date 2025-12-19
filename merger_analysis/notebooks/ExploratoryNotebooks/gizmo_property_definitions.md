# Gizmo Analysis Property Definitions

This document provides detailed information about key properties in `gizmo_analysis` for FIRE-2 simulation data.

## Quick Reference

Based on the source code at `/Users/kadofong/work/projects/merian/external/gizmo_analysis/gizmo_analysis/gizmo_io.py`

---

## 1. `form.scalefactor`

**Definition**: Expansion scale-factor when a star particle formed

**Source**:
- HDF5 field: `'StellarFormationTime'` (line 1713)
- Stored as: `'form.scalefactor'` (line 1713)

**Units**:
- **Cosmological simulations**: Scale-factor [0 to 1]
- **Non-cosmological simulations**: Time [Gyr] (line 2188)

**Location in code**: [gizmo_io.py:1713](../../external/gizmo_analysis/gizmo_analysis/gizmo_io.py#L1713)

**Unit conversion** (line 2186-2188):
```python
if 'form.scalefactor' in part[spec_name]:
    if not header['cosmological']:
        part[spec_name]['form.scalefactor'] *= time_conversion  # convert to [Gyr]
```

**Documentation** (line 91):
> Star particles also have:
>   'form.scalefactor' : expansion scale-factor when the star particle formed [0 to 1]

**Derived properties using `form.scalefactor`**:

You can compute derived properties using the `.prop()` method:

- **Formation time** (line 527-531):
  ```python
  part['star'].prop('form.time')  # Age of universe when formed [Gyr]
  # Computed as: Cosmology.get_time(form.scalefactor, 'scalefactor')
  ```

- **Formation redshift** (line 533-534):
  ```python
  part['star'].prop('form.redshift')  # Redshift when formed
  # Computed as: 1 / form.scalefactor - 1
  ```

- **Formation snapshot** (line 535-546):
  ```python
  part['star'].prop('form.snapshot')  # Snapshot index immediately after formation
  # Uses Snapshot.get_snapshot_indices() with padded scalefactor
  ```

---

## 2. `mass`

**Definition**: Particle mass

**Source**:
- HDF5 field: `'Masses'` (line 1670)
- Stored as: `'mass'` (line 1670)

**Units**: Solar masses [M_sun]

**Location in code**: [gizmo_io.py:1670](../../external/gizmo_analysis/gizmo_analysis/gizmo_io.py#L1670)

**Unit conversion** (line 2131, 2149-2152):
```python
mass_conversion = 1e10 / header['hubble']  # multiple by this for [M_sun]

for prop_name in ['mass', 'mass.bh', 'mass.disk']:
    if prop_name in part[spec_name]:
        # convert to [M_sun]
        part[spec_name][prop_name] *= mass_conversion
```

**Documentation** (line 69):
> All particle species have the following properties:
>     'mass' : mass [M_sun]

**Notes**:
- Available for all particle species: `'dark'`, `'dark2'`, `'gas'`, `'star'`, `'blackhole'`
- For black holes, `'mass'` is renamed to `'mass.total'` to avoid confusion with `'mass.bh'` (line 2154-2156)

**Derived properties using `mass`**:

- **Elemental mass** (line 362-366):
  ```python
  part['star'].prop('mass.oxygen')  # Mass in oxygen
  # Computed as: mass * massfraction.oxygen
  ```

- **Formation mass** (for stars with mass loss) (line 354-357):
  ```python
  part['star'].prop('form.mass')  # Initial formation mass
  # Computed accounting for stellar mass loss
  ```

---

## 3. `massfraction`

**Definition**: Fraction of particle mass in different elemental abundances

**Source**:
- HDF5 field: `'Metallicity'` (line 1709)
- Stored as: `'massfraction'` (line 1709)

**Units**: Linear mass fraction [dimensionless, 0 to 1]

**Location in code**: [gizmo_io.py:1709](../../external/gizmo_analysis/gizmo_analysis/gizmo_io.py#L1709)

**Data structure**: NumPy array with shape `(n_particles, n_elements)`

**Element indices** (line 1707-1708):
```
0  = all metals (everything not H, He)
1  = He (Helium)
2  = C  (Carbon)
3  = N  (Nitrogen)
4  = O  (Oxygen)
5  = Ne (Neon)
6  = Mg (Magnesium)
7  = Si (Silicon)
8  = S  (Sulfur)
9  = Ca (Calcium)
10 = Fe (Iron)
```

**Documentation** (line 85-88):
> Star particle and gas cells also have:
>     'massfraction' : fraction of the mass that is in different elemental abundances,
>         stored as an array for each particle, with indexes as follows:
>         0 = all metals (everything not H, He)
>         1 = He, 2 = C, 3 = N, 4 = O, 5 = Ne, 6 = Mg, 7 = Si, 8 = S, 9 = Ca, 10 = Fe

**Unit conversion**: No conversion for basic mass fractions, but element-tracer weights are adjusted (line 2162-2173):
```python
if ('massfraction' in part[spec_name]
    and 'ElementTracer' in part[spec_name].__dict__
    and part[spec_name].ElementTracer is not None):
    # Element-tracer mass weights converted to dimensionless units
    elementtracer_index_start = part[spec_name].ElementTracer['element.index.start']
    part[spec_name]['massfraction'][:, elementtracer_index_start:] /= mass_conversion
```

**Accessing mass fractions**:

Direct access (line 726-728):
```python
# Get oxygen mass fraction for all particles
oxygen_index = 4
oxygen_massfrac = part['star']['massfraction'][:, oxygen_index]

# Or for specific indices
oxygen_massfrac = part['star']['massfraction'][indices, oxygen_index]
```

**Derived properties using `massfraction`**:

- **Individual element mass fractions** (line 381-382):
  ```python
  part['star'].prop('massfraction.oxygen')   # Oxygen mass fraction
  part['star'].prop('massfraction.iron')     # Iron mass fraction
  part['star'].prop('massfraction.metals')   # Total metals (index 0)
  ```

- **Hydrogen mass fraction** (line 700-708):
  ```python
  part['star'].prop('massfraction.hydrogen')
  # Computed as: 1 - massfraction.helium - massfraction.metals
  ```

- **Metallicity** (log scale):
  ```python
  part['star'].prop('metallicity.iron')  # [Fe/H] in solar units
  part['star'].prop('metallicity.oxygen')  # [O/H] in solar units
  # Converted to log10(mass_fraction / mass_fraction_solar)
  # Using Asplund et al 2009 solar abundances
  ```

- **Element-tracer abundances** (line 730-762):
  ```python
  part['gas'].prop('massfraction.elementtracer.oxygen')
  # Uses element-tracer weights to compute enrichment history
  ```

---

## Snapshot-level Properties

The `scalefactor` also appears in snapshot-level metadata:

**Location**: `part.snapshot['scalefactor']`

**Definition** (line 1130-1148):
```python
if header['cosmological']:
    part.snapshot = {
        'index': snapshot_index,
        'redshift': header['redshift'],
        'scalefactor': header['scalefactor'],  # From header['time']
        'time': time,
        'time.lookback': part.Cosmology.get_time(0) - time,
        'time.hubble': header['time.hubble'],
    }
else:
    part.snapshot = {
        'index': snapshot_index,
        'redshift': 0,
        'scalefactor': 1.0,  # Always 1.0 for non-cosmological
        'time': header['time'],
        'time.lookback': 0,
        'time.hubble': None,
    }
```

**Header conversion** (line 1587-1594):
```python
if header['cosmological']:
    header['scalefactor'] = float(header['time'])  # Scale-factor from HDF5 'time' field
    del header['time']
else:
    header['scalefactor'] = 1.0  # Set to 1.0 for non-cosmological runs
```

---

## Usage Examples

```python
import gizmo_analysis as gizmo

# Load snapshot
part = gizmo.io.Read.read_snapshots(['star'], 'index', 400,
                                    simulation_directory,
                                    assign_hosts=True)

# Access basic properties
masses = part['star']['mass']  # [M_sun]
positions = part['star']['position']  # [kpc comoving]
formation_scalefactors = part['star']['form.scalefactor']  # [0-1]

# Access mass fractions
oxygen_fractions = part['star']['massfraction'][:, 4]  # Direct indexing
# Or use derived property
oxygen_fractions = part['star'].prop('massfraction.oxygen')

# Compute derived properties
formation_times = part['star'].prop('form.time')  # [Gyr]
formation_redshifts = part['star'].prop('form.redshift')
metallicities = part['star'].prop('metallicity.iron')  # [Fe/H]

# Elemental masses
oxygen_masses = part['star'].prop('mass.oxygen')  # [M_sun]

# Snapshot properties
snapshot_scalefactor = part.snapshot['scalefactor']
snapshot_redshift = part.snapshot['redshift']
```

---

## References

- Source file: `/Users/kadofong/work/projects/merian/external/gizmo_analysis/gizmo_analysis/gizmo_io.py`
- Main property dictionary: lines 1665-1720
- Unit conversions: lines 2130-2210
- Derived property parsing: lines 300-800
