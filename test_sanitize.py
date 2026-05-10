import re

def sanitize(fmt):
    parts = fmt.split("%%")
    sanitized_parts = []
    
    for p in parts:
        p = re.sub(r'%(?:[-+ #0\'I]*)(?:[0-9*]*)(?:\.[0-9*]*)?(?:hh|h|ll|l|j|z|t|L)*n', lambda m: '%' + m.group(0), p)
        sanitized_parts.append(p)
        
    return "%%".join(sanitized_parts)

print(sanitize("%-10.5lln"))
print(sanitize("%*.*n"))
print(sanitize("%n"))
print(sanitize("%%n"))
