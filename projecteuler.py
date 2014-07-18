import time

# 2
def evenFibs(n):
  from operator import add
  fibs = [0, 1]
  while fibs[-1] <= n:
    fibs.append(fibs[-1] + fibs[-2])
  evens = filter((lambda x: x % 2 == 0), fibs[:-1])
  return reduce(add, evens)

#print("evenFibs(4000000)", evenFibs(4000000))

# 4
def largestPalindromeProduct():
  palindromes = []

  def isPalindrome(n):
    return str(n) == str(n)[::-1]

  for x in range(1000):
    for y in range(1000):
      number = x * y
      if isPalindrome(number):
        palindromes.append(number)
  return max(palindromes)

#print("largestPalindromeProduct()", largestPalindromeProduct())


# 5
def smallestMultiple(n):

  def primeFactors(n):
    factors = {}
    for x in range(2, n // 2 + 1):
      while n % x == 0:
        try:
          factors[x] += 1
        except KeyError:
          factors[x] = 1
        n /= x
    if len(factors) < 1:
      return {n: 1}
    else:
      return factors
  
  multiples = {}
  def mergeFactors(dict):
    for k, v in dict.items():
      if k not in multiples:
        multiples[k] = v
      else:
        multiples[k] = max(multiples[k], v)

  # calculate multiples
  for x in range(2, n+1):
    mergeFactors(primeFactors(x))
  # multiply multiples
  f = lambda prev, curr: prev * (curr[0]**curr[1])
  return reduce(f, multiples.items(), 1)

#print("smallestMultiple(20)", smallestMultiple(20))

# 6
def sumSquareDifference(n):
  sumSquares = 0
  squareSums = 0
  for x in range(1, n+1):
    sumSquares += x**2
    squareSums += x
  squareSums **= 2
  return squareSums - sumSquares
    
#print("sumSquareDifference(100)", sumSquareDifference(100))

# 8
series = 7316717653133062491922511967442657474235534919493496983520312774506326239578318016984801869478851843858615607891129494954595017379583319528532088055111254069874715852386305071569329096329522744304355766896648950445244523161731856403098711121722383113622298934233803081353362766142828064444866452387493035890729629049156044077239071381051585930796086670172427121883998797908792274921901699720888093776657273330010533678812202354218097512545405947522435258490771167055601360483958644670632441572215539753697817977846174064955149290862569321978468622482839722413756570560574902614079729686524145351004748216637048440319989000889524345065854122758866688116427171479924442928230863465674813919123162824586178664583591245665294765456828489128831426076900422421902267105562632111110937054421750694165896040807198403850962455444362981230987879927244284909188845801561660979191338754992005240636899125607176060588611646710940507754100225698315520005593572972571636269561882670428252483600823257530420752963450
series = [int(i) for i in str(series)]
def largestProductInSeries(digits):
  from operator import mul
  return max([reduce(mul,series[i:i+digits],1) for i in range(0,len(series)-digits)])

#print("largestProductInSeries(13)", largestProductInSeries(13))

# 9
def pythagTriplet(n):
  for a in range(1, n//3):
    for b in range(a, n//2):
      c = (a**2 + b**2)**0.5
      if a + b + c == n:
        return a * b * c

#print("pythagTriplet(1000)", pythagTriplet(1000))


# 10
def primes():
  composites = {}
  i = 2
  while True:
    # if i not in composites, it's prime
    if i not in composites:
      yield i
      # add next multiple to composites
      composites[i * i] = [i]
    # i in composites
    else:
      # mark next multiples
      for prime in composites[i]:
        composites.setdefault(i + prime, []).append(prime)
      # reached i, so no longer need it
      del composites[i]      
    i += 1

def sumPrimes(n):
  total, i, gen = 0, 0, primes()
  while i < n:
    total += i
    i = next(gen)
  return total

#print("sumPrimes(2000000)", sumPrimes(2000000))

# 12
def factorize(n):
  factors = [1]
  i = 2
  while i <= n**0.5:
    if n % i == 0:
      factors.append(i)
      factors.append(n//i)
    i += 1
  factors.append(n)
  return factors

def divisibleTriangular(n):
  from operator import add
  i, triangular = 0, 0
  while True:
    triangular += i
    factors = factorize(triangular)
    if len(factors) > n:
      return triangular
    i += 1

#print("divisibleTriangular(500)", divisibleTriangular(500))


# 14
def longestCollatz(n):
  val, longest, i = 0, 0, 2
  sequences = {1: 1}
  while i < n:
    x = i
    to_add = []
    while (x != 1) and (x not in sequences):
      to_add.append(x)
      if x % 2 == 0:
        x /= 2
      else:
        x = 3 * x + 1

    length = sequences[x]
    # add to sequences
    for index in range(len(to_add)):
      sequences[to_add[index]] = len(to_add) - index + length

    # update longest and increment iteration count
    length = sequences[i]
    if length > longest:
      print("NEW LONGEST", i, length)
      val, longest = i, length
    i += 1

  return val, longest

#print("longestCollatz(1000000)", longestCollatz(1000000))

# 15
def latticePaths(n, m):
  solutions = {
    (1, 1): 0,
    (1, 2): 1,
    (2, 1): 1,
  }
  for i in range(2, n + 1):
    for j in range(2, m + 1):
      solution = 0
      # go right
      if i - 1 > 1:
        solution += solutions[(i - 1, j)]
      else:
        solution += 1
      # go down
      if j - 1 > 1:
        solution += solutions[(i, j - 1)]
      else:
        solution += 1
      # record solution
      solutions[(i, j)] = solution
  #return solutions
  return solutions[(n, m)]

#print("latticePaths(21)", latticePaths(21, 21))

# 16
def powerDigitSum(n):
  from operator import add
  digit, i = 2, 1
  powers = {1: 2}
  # build powers of 2
  while i < n:
    digit **= 2
    i *= 2
    powers[i] = digit
  # divide powers of 2
  diff = i - n
  while diff > 0:
    m = diff
    while m not in powers:
      m //= 2
    digit /= powers[m]
    diff -= m
  # sum digits
  digits = [int(x) for x in str(digit)]
  return reduce(add, digits, 0)

#print("powerDigitSum(1000)", powerDigitSum(1000))


# 17
onesWords = {'0': '', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
tensWords = {'0': '', '1': '', '2': 'twenty', '3': 'thirty', '4': 'forty', '5': 'fifty', '6': 'sixty', '7': 'seventy', '8': 'eighty', '9': 'ninety'}
teensWords = {'10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen', '18': 'eighteen', '19': 'nineteen'}
def numberToWords(n):
  num = str(n)
  words = ''

  # special case: 1000
  if len(num) == 4:
    return 'one thousand'

  # 100's place
  if len(num) >= 3:
    x = num[-3]
    words += onesWords[x] + ' hundred'
    if not (num[-2:] == '00'):
      words += ' and '

  # 10's place
  if len(num) >= 2:
    x = num[-2]
    # special cases: 10-19
    if x == '1':
      x = num[-2:]
      words += teensWords[x]
      return words
    else:
      words += tensWords[x]
      if not (num[-1] == '0' or x == '0'):
        words += '-'

  # 1's place
  if len(num) >= 1:
    x = num[-1]
    words += onesWords[x]

  return words

def numberLetterCounts(n):
  from re import sub
  count = 0
  for x in range(1,n+1):
    #print(x, numberToWords(x))
    letters = sub(r'[ ]*[-]*', '', numberToWords(x))
    count += len(letters)
  return count

#print("numberLetterCounts(1000)", numberLetterCounts(1000))

# 24
def lexicographicPermutations(n):

  digits = '0123456789'
  # init solution space with base solution
  solutions = {'0': ['0']}

  # populate solution space bottom-up
  for i in range(1, len(digits)):
    digit = digits[i]
    subProblem = digits[:i+1]
    prevSolution = solutions[digits[:i]]
    solution = []
    # add newest digit to each possible location
    for prev in prevSolution:
      for x in range(len(prev)+1):
        solution.append(prev[:x] + digit + prev[x:])

    # add new solution
    solutions[subProblem] = solution

  return sorted(solutions[digits])[n-1]

#print("lexicographicPermutations(10**6)", lexicographicPermutations(10**6))

# 20
def factorialDigitSum(n):
  from operator import mul, add
  factorial = reduce(mul, range(1, n + 1), 1)
  digits = [int(i) for i in str(factorial)]
  return reduce(add, digits, 0)

#print('factorialDigitSum(100)', factorialDigitSum(100))


# 21
def factorize(n):
  if n == 1:
    return []
  factors = [1]
  i = 2
  while i * i <= n:
    if n % i == 0:
      factors.append(i)
      if not (n / i == i):
        factors.append(n / i)
    i += 1
  return factors

def amicableSum(n):
  from operator import add

  # calculate factor sums
  factorSums = [sum(factorize(i)) for i in range(n)]

  # calculate amicables
  amicables = set()
  for i in range(1, n):
    for j in range(i, n-1):
      if (i == factorSums[j]) and (j == factorSums[i]) and (not (i == j)):
        amicables.update([i, j])
  #return amicables
  return sum(amicables)

#start = time.time()
#ans = amicableSum(10**4)
#elapsed = time.time() - start
#print("%s: %s found in %s seconds") % ('amicableSum', ans, elapsed)


# 22
def mergeSort(lst):
  if len(lst) < 2:
    return lst
  else:
    half = len(lst) // 2
    left = mergeSort(lst[:half])
    right = mergeSort(lst[half:])
    result = []
    i, j = 0, 0
    while (i < len(left)) and (j < len(right)):
      if left[i] < right[j]:
        result.append(left[i])
        i += 1
      else:
        result.append(right[j])
        j += 1
    return result + left[i:] + right[j:]

def nameScore(name):
  base = ord('A') - 1
  fn = lambda accum, curr: accum + (ord(curr) - base)
  return reduce(fn, name, 0)

def namesScores():
  fname = '/Users/Bell/projects/projecteuler/names.txt'
  with open(fname, 'r') as f:
    names = f.read().replace('"', '').split(',')
    names = mergeSort(names)
    return sum([nameScore(names[i]) * (i + 1) for i in range(len(names))])

start = time.time()
ans = namesScores()
elapsed = time.time() - start
print("%s: %s found in %s seconds") % ('namesScores', ans, elapsed)


class BinaryHeapTree(object):
  def __init__(self, entry=None, left=None, right=None):
    self.lastInsert = 0
    self.entry = entry
    self.left = left
    self.right = right

  def insert(self, entry):
    if not self.left:
      self.left = BinaryHeapTree(entry)
    elif not self.right:
      self.right = BinaryHeapTree(entry)
    else:
      if self.lastInsert == 0:
        self.left.insert(entry)
      else:
        self.right.insert(entry)
      self.lastInsert = (self.lastInsert + 1) % 2

  def height(self):
    leftHeight = self.left.height(h) if self.left else 0
    rightHeight = self.right.height(h) if self.right else 0
    return 1 + max(leftHeight, rightHeight)

class MinHeap(object):
  def __init__(self, seq=[]):
    if seq:
      # TODO: init from seq
      Build-Max-Heap (A):
        heap_length[A] ← length[A]
        for i ← floor(length[A]/2) downto 1 do
          Max-Heapify(A, i)
          
      self.tree = None
    else:
      self.tree = None
      
  def pop(self):
    tree = self.tree
    if tree:
      if not tree.left:
        self.tree = tree.right
      elif not tree.right:
        self.tree = tree.left
      elif tree.left.entry < tree.right.entry:
        self.tree = tree.left
      else:
        self.tree = tree.right
      return tree.entry
    else:
      return None

  def peek(self):
    return self.tree.entry if self.tree else None

  def insert(self, entry):
    if self.tree:
      self.tree.insert(entry)
    else:
      self.tree = BinaryHeapTree(entry)



def heapSort(seq):
  for i in range(len(seq)):
    seq[i] = heap.pop()
  return seq























