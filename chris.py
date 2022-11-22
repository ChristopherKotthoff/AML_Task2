import inspect

def add( a,b = 6, **args ):
  pass

print(str(inspect.signature(add)))
print(add.__name__)
