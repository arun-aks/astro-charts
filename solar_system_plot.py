"""Library of functions to generate star charts.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from skyfield.api import load, Star, utc
from skyfield.data import hipparcos, stellarium


def plot_solar_sys_objects(ax, datetime=None, limiting_magnitude=3.0,
                          ecliptic_coords=True):
  """Generate a chart (plot) of solar system objects against background stars.

  Args:
      ax (matplotlib axis): Axis to draw the chart on.

      datetime (datetime object, optional): Specify the date/time for which the
      chart must be generated. Defaults to None (current system time).

      limiting_magnitude (float, optional): Stars of magnitude above this (i.e.,
      fainter) are ignored. Defaults to 3.0.

      ecliptic_coords (bool, optional): If True, ecliptic coordinates are used
      (i.e., x-axis is the ecliptic). If False, RA-declination coordinates are
      used.

  Returns:
      None
  """
  ts = load.timescale()
  if datetime is None:
      t = ts.now()
  else:
      t = ts.from_datetime(datetime.replace(tzinfo=utc))

  # An ephemeris from the JPL provides Solar System Object Positions.
  eph = load('de421.bsp')
  earth = eph['earth']

  # Load star data from Hipparcos catalog
  with load.open(hipparcos.URL) as f:
    stars = hipparcos.load_dataframe(f)

  # And the constellation outlines come from Stellarium.  We make a list
  # of the stars at which each edge stars, and the star at which each edge
  # ends.
  url = ('https://raw.githubusercontent.com/Stellarium/stellarium/master'
          '/skycultures/modern_st/constellationship.fab')
  with load.open(url) as f:
    constellations = stellarium.parse_constellations(f)

  edges = [edge for name, edges in constellations for edge in edges]
  edges_star1 = [star1 for star1, star2 in edges]
  edges_star2 = [star2 for star1, star2 in edges]

  # Create a True/False mask marking the stars bright enough to be
  # included in our plot.
  bright_stars = (stars.magnitude <= limiting_magnitude)

  # Add stars on the edges irrespective of their brightness
  bright_stars[edges_star1] = True
  bright_stars[edges_star2] = True

  # And go ahead and compute how large their markers will be on the plot.
  magnitude = stars['magnitude'][bright_stars]
  marker_size = (1.5 + limiting_magnitude - magnitude) ** 2.0

  # Get the star positions as seen from earth at the specified time
  star_positions = earth.at(t).observe(Star.from_dataframe(stars))

  # Plot stars using ecliptic coordinates
  if ecliptic_coords:
    ecl_lat, ecl_lon = star_positions.ecliptic_latlon()[0:2]
    stars['lat_deg'] = ecl_lat.degrees
    stars['lon_deg'] = ecl_lon.degrees
  else:
    ra, dec = star_positions.radec()[:2]
    stars['lat_deg'] = dec.degrees
    stars['lon_deg'] = ra.hours

  ax.scatter(stars['lon_deg'][bright_stars],
            stars['lat_deg'][bright_stars],
            s=marker_size, color='#fff')

  xy1 = stars[['lon_deg', 'lat_deg']].loc[edges_star1].values
  xy2 = stars[['lon_deg', 'lat_deg']].loc[edges_star2].values

  # xy1 and xy2 are (N, 2) arrays. So, [xy1, xy2] will be (2, N, 2) array.
  # We want to roll the axes to get (N, 2, 2) shape array, so the final axis
  # is the one that specifies the end points xy1 and xy2.
  lines_xy = np.rollaxis(np.array([xy1, xy2]), 1)

  lines_xy_mod = []

  def dist(a, b):
    return np.sqrt(np.sum((a - b)**2.0))

  x_max = 360 if ecliptic_coords else 24

  # Add stars beyond 0 to x_max longitude so that constellation lines don't
  # run in the wrong direction
  for line in lines_xy:
    if dist(line[0], line[1]) > dist(line[0] + (x_max, 0), line[1]):
      lines_xy_mod.append([line[0] + (x_max, 0), line[1]])
      lines_xy_mod.append([line[0], line[1] - (x_max, 0)])
    elif dist(line[0], line[1]) > dist(line[0], line[1] + (x_max, 0)):
      lines_xy_mod.append([line[0] - (x_max, 0), line[1]])
      lines_xy_mod.append([line[0], line[1] + (x_max, 0)])
    else:
      lines_xy_mod.append([line[0], line[1]])

  lines_xy_mod = np.array(lines_xy_mod)

  ax.set_facecolor((0.1, 0.1, 0.3))
  ax.add_collection(
    LineCollection(lines_xy_mod, colors='#fff', linewidth=0.25))
  ax.set_ylim(-30, 30)
  ax.set_xlim(x_max, 0) # Flip to get the correct view.
  ax.set_aspect(x_max/360)  # Lock aspect ratio so the stars don't look squished.
  ax.grid(linewidth=0.3, color='#00f')

  # Key = Object name in eph, value[0] = marker color,
  # value[1] = printed name
  objs = {'sun': ['yellow', 10, 'Sun'],
          'moon': ['grey', 10, 'Moon'],
          'mercury': ['orange', 6, 'Mercury'],
          'venus': ['silver', 6, 'Venus'],
          'mars': ['red', 5, 'Mars'],
          'jupiter_barycenter': ['brown', 5, 'Jupiter'],
          'saturn_barycenter': ['lightgrey', 5, 'Saturn'],
          'uranus_barycenter': ['cyan', 4, 'Uranus'],
          'neptune_barycenter': ['blue', 4, 'Neptune']}

  # Show the ecliptic
  if ecliptic_coords:
    ax.axhline(0, linewidth=1, color='#55f')
  else:
    line = []
    for ofst in range(-190, 190):
      obj_pos = earth.at(t+ofst).observe(eph['sun'])
      ra, dec = obj_pos.radec()[:2]
      x = ra.hours
      y = dec.degrees
      line.append([x, y])

    line = np.array(line)
    idx_arr = np.argsort(line[:, 0])
    ax.plot(line[idx_arr, 0], line[idx_arr, 1], linewidth=1, color='#55f')


  # Plot the position of the solar system objects
  for obj in objs:
    obj_eph = eph[obj]
    color, ms, name = objs[obj]
    obj_pos = earth.at(t).observe(obj_eph)

    if ecliptic_coords:
      ecl_lat, ecl_lon = obj_pos.ecliptic_latlon()[:2]
      x = ecl_lon.degrees
      y = ecl_lat.degrees
    else:
      ra, dec = obj_pos.radec()[:2]
      x = ra.hours
      y = dec.degrees

    ax.plot(x, y, marker='o', linestyle=' ', markersize=ms,
            label=name, markerfacecolor=color, markeredgecolor=color)
    ax.text(x, y, ' ' + name, va='bottom', rotation=90, clip_on=True,
            color='w')

    obj_pos1 = earth.at(t+7).observe(obj_eph)

    if ecliptic_coords:
      ecl_lat1, ecl_lon1 = obj_pos1.ecliptic_latlon()[:2]

      lat_delta = ecl_lat1.degrees - ecl_lat.degrees
      lon_delta = ecl_lon1.degrees - ecl_lon.degrees
    else:
      ra1, dec1 = obj_pos1.radec()[:2]

      lat_delta = dec1.degrees - dec.degrees
      lon_delta = ra1.hours - ra.hours

    x1 = x + lon_delta
    y1 = y + lat_delta

    if obj in ['mercury', 'venus', 'sun']:
      ax.annotate('', [x, y], [x1, y1],
                  arrowprops={'arrowstyle': '<-', 'color': 'w'})

  if ecliptic_coords:
    ax.set_xlabel('Ecliptic Latitude (degrees)')
    ax.set_ylabel('Ecliptic Longitude (degrees)')
  else:
    ax.set_xlabel('Right Ascension (hours)')
    ax.set_ylabel('Declination (degrees)')

  ax.set_title('Sun, Moon and Planets at ' + f'{t.utc_strftime()}')

  # Set xticks
  if ecliptic_coords:
    ax.set_xticks(np.arange(0, 361, 30))
  else:
    ax.set_xticks(np.arange(0, 25, 2))

  return


if __name__ == "__main__":
  fh, axs = plt.subplots(2, 1, figsize=[19, 9])
  plot_solar_sys_objects(axs[0])
  plot_solar_sys_objects(axs[1], ecliptic_coords=False)
  axs[0].legend(framealpha=0.1, labelcolor='w')
  fh.set_tight_layout(True)
