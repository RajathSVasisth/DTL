from Crypto.Cipher import AES 
from Crypto import Random
import binascii
#----------------------------------------------------------------------------------------------------#
Pcurve = 2**256 - 2**32 - 2**9 - 2**8 - 2**7 - 2**6 - 2**4 -1
N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141 
Acurve = 0; Bcurve = 7 
Gx = 55066263022277343669578718895168534326250603453777594175500187360389116729240
Gy = 32670510020758816978083085130507043184471273380659243275938904335757337482424
GPoint = (Gx,Gy)
#----------------------------------------------------------------------------------------------------#
publicKey = 0xA0DC65FFCA799873CBEA0AC274015B9526505DAAAED385155425F7337704883E
#----------------------------------------------------------------------------------------------------#
#Extended Euclidean Algorithm
def exeuclid(a, n=Pcurve):
    low1, high1 = 1,0
    low2, high2 = a%n,n
    while low2 > 1:
        ratio = high2//low2
        nm, new = high1-low1*ratio, high2-low2*ratio
        low1, low2, high1, high2 = nm, new, low1, low2
    return low1 % n
#----------------------------------------------------------------------------------------------------#
#Addition on Finite Field Group on the Elliptic Curve
def ECCadd(a,b):
    Lambda = ((b[1]-a[1])*exeuclid(b[0]-a[0],Pcurve))%Pcurve
    x = (Lambda**2 - a[0]-b[0])%Pcurve
    y = (Lambda*(a[0]-x)-a[1])%Pcurve
    return(x,y)
#----------------------------------------------------------------------------------------------------#
#Doubling of a point on the Elliptic Curve
def ECCdouble(a):
    lamdub = ((3*a[0]*a[0]+Acurve)*exeuclid((2*a[1]),Pcurve))%Pcurve
    x = (lamdub**2-2*a[0])%Pcurve
    y = (lamdub*(a[0]-x)-a[1])%Pcurve
    return(x,y)
#----------------------------------------------------------------------------------------------------#
#Main Function to multiply the point by a scalar, n times, to obtain the final key
def ECCMultiply(Generator, ScalarHexa):
    if ScalarHexa==0 or ScalarHexa>=N:
        raise Exception("Invalid Private key")
    ScalarBinary = str(bin(ScalarHexa))[2:]
    Q=Generator
    for i in range(1, len(ScalarBinary)):
        Q = ECCdouble(Q)
        if ScalarBinary[i]=="1":
            Q = ECCadd(Q, Generator)
    return(Q)
#----------------------------------------------------------------------------------------------------#
PrivateKey = ECCMultiply(GPoint, publicKey)
privateKeyfinal = str(hex(PrivateKey[0])[2:-1]).zfill(64)

#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
def bytes_to_int(bytes):
    result = 0
    for b in bytes:
        result = result * 256 + int(b)
    return result
#----------------------------------------------------------------------------------------------------#
def int_to_bytes(value, length):
    result = []

    for i in range(0, length):
        result.append(value >> (i * 8) & 0xff)

    result.reverse()

    return result
#----------------------------------------------------------------------------------------------------#
def append_space_padding(str, blocksize = 128):
    pad_len = blocksize - (len(str) % blocksize)
    padding = 'a'*pad_len
    return str + padding
#----------------------------------------------------------------------------------------------------#
def remove_space_padding(str, blocksize=128):
    pad_len = 0
    for char in str[::-1]:
        if char == 'a':
            pad_len += 1
        else:
            break
    str = str[:-pad_len]
    return str
#----------------------------------------------------------------------------------------------------#
def encrypt(plaintext, key):
    aes = AES.new(key, AES.MODE_ECB)
    return aes.encrypt(plaintext)
#----------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------#

def arguement_taker(plain_text):
    #privkey = Random.new().read(16)
     publicKey = privateKeyfinal
     publicKey = binascii.unhexlify(publicKey)
     plaintext = plain_text
     #print(len(plaintext))
     plaintext = append_space_padding(plaintext)
     #print(len(plaintext))
     #print(plaintext)

     #ciphertext = encrypt(plaintext, binascii.unhexlify(bytearray(public_key, encoding = 'UTF-8')))
     ciphertext = encrypt(plaintext, publicKey)
     crypt = binascii.hexlify(bytearray(ciphertext))
     return crypt


def decrypter(cipher_text):
    publicKey = privateKeyfinal
    publicKey = binascii.unhexlify(publicKey)
    byte_text = binascii.unhexlify(cipher_text)
    aes = AES.new(publicKey, AES.MODE_ECB)
    decrypted = aes.decrypt(byte_text).decode('ISO-8859-1')
    decrypted = remove_space_padding(decrypted)
    return decrypted









