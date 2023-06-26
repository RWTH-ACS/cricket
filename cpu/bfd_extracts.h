/* DO NOT EDIT!  -*- buffer-read-only: t -*-  This file is automatically 
   generated from "libbfd-in.h", "init.c", "libbfd.c", "bfdio.c", 
   "bfdwin.c", "cache.c", "reloc.c", "archures.c" and "elf.c".
   Run "make headers" in your build bfd/ to regenerate.  */

/* libbfd.h -- Declarations used by bfd library *implementation*.
   (This include file is not for users of the library.)

   Copyright 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998,
   1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
   2010, 2011, 2012
   Free Software Foundation, Inc.

   Written by Cygnus Support.

   This file is part of BFD, the Binary File Descriptor library.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street - Fifth Floor, Boston,
   MA 02110-1301, USA.  */
   
#include <bfd.h>
/* Extracted from bfdio.c.  */
struct bfd_iovec
{
  /* To avoid problems with macros, a "b" rather than "f"
     prefix is prepended to each method name.  */
  /* Attempt to read/write NBYTES on ABFD's IOSTREAM storing/fetching
     bytes starting at PTR.  Return the number of bytes actually
     transfered (a read past end-of-file returns less than NBYTES),
     or -1 (setting <<bfd_error>>) if an error occurs.  */
  file_ptr (*bread) (struct bfd *abfd, void *ptr, file_ptr nbytes);
  file_ptr (*bwrite) (struct bfd *abfd, const void *ptr,
                      file_ptr nbytes);
  /* Return the current IOSTREAM file offset, or -1 (setting <<bfd_error>>
     if an error occurs.  */
  file_ptr (*btell) (struct bfd *abfd);
  /* For the following, on successful completion a value of 0 is returned.
     Otherwise, a value of -1 is returned (and  <<bfd_error>> is set).  */
  int (*bseek) (struct bfd *abfd, file_ptr offset, int whence);
  int (*bclose) (struct bfd *abfd);
  int (*bflush) (struct bfd *abfd);
  int (*bstat) (struct bfd *abfd, struct stat *sb);
  /* Mmap a part of the files. ADDR, LEN, PROT, FLAGS and OFFSET are the usual
     mmap parameter, except that LEN and OFFSET do not need to be page
     aligned.  Returns (void *)-1 on failure, mmapped address on success.
     Also write in MAP_ADDR the address of the page aligned buffer and in
     MAP_LEN the size mapped (a page multiple).  Use unmap with MAP_ADDR and
     MAP_LEN to unmap.  */
  void *(*bmmap) (struct bfd *abfd, void *addr, bfd_size_type len,
                  int prot, int flags, file_ptr offset,
                  void **map_addr, bfd_size_type *map_len);
};