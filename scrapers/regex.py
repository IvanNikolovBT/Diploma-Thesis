import re
test_string="Македонска поезија: „Сенките нè одминуваат“ од Никола Маџиров"

pattern = r'Македонска поезија: „(.+?)“ од ([\w\s]+)'
matches = re.match(pattern, test_string)
print(matches[0])
print(matches[1])
print(matches[2])