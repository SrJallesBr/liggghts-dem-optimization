if "${rolling_event_saved} == 0 && ${rolling_event_crossed} == 1" then &
  "write_dump all custom post/rolling_event_particles.dump id type x y z radius vx vy vz omegax omegay omegaz modify sort id format '%d %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g'" &
  "write_dump all mesh/stl post/rolling_event_base_*.stl base" &
  "print ROLLING_EVENT_V7_PRESENT_POPULATION,step=$(step),angle=${angulo_gravidade_graus},moving=$(c_moving_count),N0=${initial_count},alive=$(count(all)) file post/rolling_event.txt screen no" &
  "variable rolling_event_saved equal 1"
