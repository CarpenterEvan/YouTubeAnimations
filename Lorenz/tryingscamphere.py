import scamp

s = scamp.Session()

violin = s.new_part("violin")
harp = s.new_part("harp")
midi = s.new_midi_part()

while True:
    violin.play_note(70, 1, 0.5)
    violin.play_note(80, 1, 1)
    violin.play_note(78, 1, 0.5)
    violin.play_note(77, 1, 1)
