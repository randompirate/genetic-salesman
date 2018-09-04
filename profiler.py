from genetic_salesman import profile_me, profile_setup
import cProfile, pstats, io
from datetime import datetime

datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

pr = cProfile.Profile()


G_prof = profile_setup()
pr.enable()
profile_me(G_prof)
pr.disable()

s = io.StringIO()
sortby = 'time'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()

with open('profile{}.txt'.format(datestamp), 'w') as f:
  f.write(s.getvalue())
