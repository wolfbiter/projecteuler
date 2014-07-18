// 1
function listMultiples(n) {
  var multiples = [];
  var i = 0;
  while (i < n) {
    if (i % 3 == 0) {
      multiples.push(i);
    } else if (i % 5 == 0) {
      multiples.push(i);
    }
    i++;
  }
  var sum = 0;
  return multiples.reduce(function (prev, curr, index, array) {
    return prev + curr;
  });
}

// 3
function largestPrimeFactor(n) {
  var i = 2;
  while (i * i < n) {
    while (n % i == 0) {
      n = n / i;
    }
    i++;
  }
  return n
}

// 7
function primes(n) {
  // init primes
  var primes = [];
  for (var x = 2; x <= n / 2.0; x++) {
    primes.push(x);
  }
  // calculate primes less than n / 2
  var i = 0;
  while (i < primes.length) {
    var prime = primes[i];
    // filter primes
    var restPrimes = primes.slice(i+1).filter(function (val) {
      return val % prime !== 0;
    });
    var temp = primes.slice(i+1);
    primes = primes.slice(0, i+1).concat(restPrimes);
    i++;
  }
  return primes;
}




