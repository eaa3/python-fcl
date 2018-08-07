from libcpp.string cimport string

cdef extern from "<iostream>" namespace "std":
    cdef cppclass istream:
        istream& write(const char*, int) except +

    cdef cppclass iostream(istream):
        iostream() except +

    cdef cppclass stringstream(iostream):
        stringstream() except +
        string str() const
        void str(string& s)
    
    cdef cppclass ostream:
        ostream& write(const char*, int) except +


cdef extern from "<sstream>" namespace "std":
    cdef cppclass istringstream:
        istringstream() except +
        istringstream(string& s) except +
        string str()
        void str(string& s)
    cdef cppclass ostringstream:
        ostringstream() except +
        string str()
        void close()

cdef extern from "<iostream>" namespace "std::ios_base":
    cdef cppclass open_mode:
        pass
    cdef open_mode binary
    cdef open_mode out 

cdef extern from "<fstream>" namespace "std":         
     cdef cppclass filebuf:
          pass            
     cdef cppclass fstream:
          void close()
          bint is_open()
          void open(const char*, open_mode)
          void open(const char&, open_mode)
          filebuf* rdbuf() const
          filebuf* rdbuf(filebuf* sb)              
     cdef cppclass ofstream(ostream):
          ofstream() except +
          ofstream(const char*) except +
          ofstream(const char*, open_mode) except+
          void open(const char*, open_mode)
          void close()
     cdef cppclass ifstream(istream):
          ifstream(const char*) except +
          ifstream(const char*, open_mode) except+              
