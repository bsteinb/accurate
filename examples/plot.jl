using Winston

function read_results(fname)
  f = open(fname)
  d = readcsv(f)
  close(f)
  x = d[:, 1]
  y = d[:, 2]
  (x, y)
end

loglog()
title("Naive Summation, double precision")
xlabel("condition number")
xlim(1e-1, 1e141)
ylabel("relative error")
ylim(1e-17, 1e1)

x, y = read_results("Naive.csv")
oplot(x, y, ".")

savefig("Naive.svg")
savefig("Naive.pdf")

loglog()
title("Comparison of SumK for K = 2 .. 9, double precision")
xlabel("condition number")
xlim(1e-1, 1e141)
ylabel("relative error")
ylim(1e-17, 1e1)

x, y = read_results("Sum2.csv")
oplot(x, y, "k.")

x, y = read_results("Sum3.csv")
oplot(x, y, "r.")

x, y = read_results("Sum4.csv")
oplot(x, y, "g.")

x, y = read_results("Sum5.csv")
oplot(x, y, "b.")

x, y = read_results("Sum6.csv")
oplot(x, y, "k.")

x, y = read_results("Sum7.csv")
oplot(x, y, "r.")

x, y = read_results("Sum8.csv")
oplot(x, y, "g.")

x, y = read_results("Sum9.csv")
oplot(x, y, "b.")

legend(Any["Sum2", "Sum3", "Sum4", "Sum5", "Sum6", "Sum7", "Sum8", "Sum9"])

savefig("SumK.svg")
savefig("SumK.pdf")

loglog()
title("OnlineExactSum, double precision")
xlabel("condition number")
xlim(1e-1, 1e141)
ylabel("relative error")
ylim(1e-17, 1e1)

x, y = read_results("OnlineExactSum.csv")
oplot(x, y, ".")

savefig("OnlineExactSum.svg")
savefig("OnlineExactSum.pdf")

loglog()
title("Comparison of DotK for K = 2 .. 9, double precision")
xlabel("condition number")
xlim(1e-1, 1e141)
ylabel("relative error")
ylim(1e-17, 1e1)

x, y = read_results("Dot2.csv")
oplot(x, y, "k.")

x, y = read_results("Dot3.csv")
oplot(x, y, "r.")

x, y = read_results("Dot4.csv")
oplot(x, y, "g.")

x, y = read_results("Dot5.csv")
oplot(x, y, "b.")

x, y = read_results("Dot6.csv")
oplot(x, y, "k.")

x, y = read_results("Dot7.csv")
oplot(x, y, "r.")

x, y = read_results("Dot8.csv")
oplot(x, y, "g.")

x, y = read_results("Dot9.csv")
oplot(x, y, "b.")

legend(Any["Dot2", "Dot3", "Dot4", "Dot5", "Dot6", "Dot7", "Dot8", "Dot9"])

savefig("DotK.svg")
savefig("DotK.pdf")

loglog()
title("OnlineExactDot, double precision")
xlabel("condition number")
xlim(1e-1, 1e141)
ylabel("relative error")
ylim(1e-17, 1e1)

x, y = read_results("OnlineExactDot.csv")
oplot(x, y, ".")

savefig("OnlineExactDot.svg")
savefig("OnlineExactDot.pdf")
