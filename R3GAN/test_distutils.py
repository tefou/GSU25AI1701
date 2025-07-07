try:
    import distutils._msvccompiler
    print("distutils._msvccompiler is available!")
except Exception as e:
    print("ERROR:", e)
