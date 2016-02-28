function DotExact(x, y)
  function DotExactAux()
    acc = BigFloat(0)

    for i = 1:size(x, 1)
      a = BigFloat(x[i])
      b = BigFloat(y[i])
      c = a * b
      acc = acc + c
    end

    convert(Float64, acc)
  end
  with_bigfloat_precision(DotExactAux, 1024)
end

function GenDot(n, c)
  n2 = round(Int, n / 2)
  x = zeros(n, 1)
  y = zeros(n, 1)

  b = log2(c)
  e = round(rand(n2, 1) * b / 2.0)
  e[1] = round(b / 2.0) + 1.0
  e[end] = 0.0
  x[1 : n2] = (2.0 * rand(n2, 1) - 1.0) .* (2.0.^e)
  y[1 : n2] = (2.0 * rand(n2, 1) - 1.0) .* (2.0.^e)

  e = round(linspace(b / 2.0, 0.0, n - n2))
  for i = n2 + 1 : n
    x[i] = (2.0 * rand() - 1.0) * 2.0^e[i - n2]
    y[i] = ((2.0 * rand() - 1.0) * 2.0^e[i - n2] - DotExact(x, y)) / x[i]
  end

  index = randperm(n)
  x = x[index]
  y = y[index]
  d = DotExact(x, y)
  C = 2.0 * DotExact(abs(x), abs(y)) / abs(d)

  (x, y, d, C[1])
end

function SumExact(z)
  function SumExactAux()
    acc = BigFloat(0)

    for i = 1 : size(z, 1)
      acc = acc + z[i]
    end

    convert(Float64, acc)
  end
  with_bigfloat_precision(SumExactAux, 1024)
end

function TwoProductFMA(a, b)
  x = a * b
  y = fma(a, b, -x)
  (x, y)
end

function GenSum(n, c)
  n2 = round(Int, n / 2)
  x, y, d, c = GenDot(n2, c)
  z = zeros(n, 1)
  for i = 1 : n2
    a, b = TwoProductFMA(x[i], y[i])
    z[2 * i - 1] = a
    z[2 * i] = b
  end
  index = randperm(n)
  z = z[index]
  s = SumExact(z)
  c = sum(abs(z)) / abs(s)
  (z, s, c)
end

fd = open("illdot.csv", "w")
fs = open("illsum.csv", "w")
emax = 140
for e in 1 : emax
  @printf("working on exponent %e of %e", e, emax)
  for i in 1 : 10
    print(".")

    x, y, d, c = GenDot(1000, 10.0^e)
    writecsv(fd, x')
    writecsv(fd, y')
    writecsv(fd, d)
    writecsv(fd, c)

    z, s, c = GenSum(2000, 10.0^e)
    writecsv(fs, z')
    writecsv(fs, s)
    writecsv(fs, c)
  end
  println(" done.")
end

