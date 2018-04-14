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
title("Naive summation, double precision")
xlabel("condition number")
xlim(1e-1, 1e281)
ylabel("relative error")
ylim(1e-17, 1e1)

x, y = read_results("NaiveSum.csv")
oplot(x, y, ".")

savefig("NaiveSum.svg")
savefig("NaiveSum.pdf")

loglog()
title("Parallel naive summation, double precision")
xlabel("condition number")
xlim(1e-1, 1e281)
ylabel("relative error")
ylim(1e-17, 1e1)

x, y = read_results("ParallelNaiveSum.csv")
oplot(x, y, ".")

savefig("ParallelNaiveSum.svg")
savefig("ParallelNaiveSum.pdf")

loglog()
title("Comparison of SumK for K = 2 .. 9, double precision")
xlabel("condition number")
xlim(1e-1, 1e281)
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

#legend(["Sum2", "Sum3", "Sum4", "Sum5", "Sum6", "Sum7", "Sum8", "Sum9"])

savefig("SumK.svg")
savefig("SumK.pdf")

loglog()
title("Comparison of parallel SumK for K = 2 .. 9, double precision")
xlabel("condition number")
xlim(1e-1, 1e281)
ylabel("relative error")
ylim(1e-17, 1e1)

x, y = read_results("ParallelSum2.csv")
oplot(x, y, "k.")

x, y = read_results("ParallelSum3.csv")
oplot(x, y, "r.")

x, y = read_results("ParallelSum4.csv")
oplot(x, y, "g.")

x, y = read_results("ParallelSum5.csv")
oplot(x, y, "b.")

x, y = read_results("ParallelSum6.csv")
oplot(x, y, "k.")

x, y = read_results("ParallelSum7.csv")
oplot(x, y, "r.")

x, y = read_results("ParallelSum8.csv")
oplot(x, y, "g.")

x, y = read_results("ParallelSum9.csv")
oplot(x, y, "b.")

#legend(["Sum2", "Sum3", "Sum4", "Sum5", "Sum6", "Sum7", "Sum8", "Sum9"])

savefig("ParallelSumK.svg")
savefig("ParallelSumK.pdf")

loglog()
title("OnlineExactSum, double precision")
xlabel("condition number")
xlim(1e-1, 1e281)
ylabel("relative error")
ylim(1e-17, 1e1)

x, y = read_results("OnlineExactSum.csv")
oplot(x, y, ".")

savefig("OnlineExactSum.svg")
savefig("OnlineExactSum.pdf")

loglog()
title("Parallel OnlineExactSum, double precision")
xlabel("condition number")
xlim(1e-1, 1e281)
ylabel("relative error")
ylim(1e-17, 1e1)

x, y = read_results("ParallelOnlineExactSum.csv")
oplot(x, y, ".")

savefig("ParallelOnlineExactSum.svg")
savefig("ParallelOnlineExactSum.pdf")

loglog()
title("Naive dot product, double precision")
xlabel("condition number")
xlim(1e-1, 1e281)
ylabel("relative error")
ylim(1e-17, 1e1)

x, y = read_results("NaiveDot.csv")
oplot(x, y, ".")

savefig("NaiveDot.svg")
savefig("NaiveDot.pdf")

loglog()
title("Parallel naive dot product, double precision")
xlabel("condition number")
xlim(1e-1, 1e301)
ylabel("relative error")
ylim(1e-17, 1e1)

x, y = read_results("ParallelNaiveDot.csv")
oplot(x, y, ".")

savefig("ParallelNaiveDot.svg")
savefig("ParallelNaiveDot.pdf")

loglog()
title("Comparison of DotK for K = 2 .. 9, double precision")
xlabel("condition number")
xlim(1e-1, 1e301)
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

#legend(["Dot2", "Dot3", "Dot4", "Dot5", "Dot6", "Dot7", "Dot8", "Dot9"])

savefig("DotK.svg")
savefig("DotK.pdf")

loglog()
title("Comparison of parallel DotK for K = 2 .. 9, double precision")
xlabel("condition number")
xlim(1e-1, 1e301)
ylabel("relative error")
ylim(1e-17, 1e1)

x, y = read_results("ParallelDot2.csv")
oplot(x, y, "k.")

x, y = read_results("ParallelDot3.csv")
oplot(x, y, "r.")

x, y = read_results("ParallelDot4.csv")
oplot(x, y, "g.")

x, y = read_results("ParallelDot5.csv")
oplot(x, y, "b.")

x, y = read_results("ParallelDot6.csv")
oplot(x, y, "k.")

x, y = read_results("ParallelDot7.csv")
oplot(x, y, "r.")

x, y = read_results("ParallelDot8.csv")
oplot(x, y, "g.")

x, y = read_results("ParallelDot9.csv")
oplot(x, y, "b.")

#legend(["Dot2", "Dot3", "Dot4", "Dot5", "Dot6", "Dot7", "Dot8", "Dot9"])

savefig("ParallelDotK.svg")
savefig("ParallelDotK.pdf")

loglog()
title("OnlineExactDot, double precision")
xlabel("condition number")
xlim(1e-1, 1e301)
ylabel("relative error")
ylim(1e-17, 1e1)

x, y = read_results("OnlineExactDot.csv")
oplot(x, y, ".")

savefig("OnlineExactDot.svg")
savefig("OnlineExactDot.pdf")

loglog()
title("Parallel OnlineExactDot, double precision")
xlabel("condition number")
xlim(1e-1, 1e301)
ylabel("relative error")
ylim(1e-17, 1e1)

x, y = read_results("ParallelOnlineExactDot.csv")
oplot(x, y, ".")

savefig("ParallelOnlineExactDot.svg")
savefig("ParallelOnlineExactDot.pdf")
