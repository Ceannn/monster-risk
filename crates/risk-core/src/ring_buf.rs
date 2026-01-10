#[derive(Debug, Clone)]
pub(crate) struct RingBuf<T: Copy + Default> {
    buf: Vec<T>,
    head: usize,
    len: usize,
}

impl<T: Copy + Default> Default for RingBuf<T> {
    fn default() -> Self {
        Self {
            buf: Vec::new(),
            head: 0,
            len: 0,
        }
    }
}

impl<T: Copy + Default> RingBuf<T> {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub(crate) fn front(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            Some(&self.buf[self.head])
        }
    }

    #[inline]
    pub(crate) fn back_mut(&mut self) -> Option<&mut T> {
        if self.len == 0 {
            return None;
        }
        let idx = (self.head + self.len - 1) % self.buf.len();
        Some(&mut self.buf[idx])
    }

    #[inline]
    pub(crate) fn pop_front(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let v = self.buf[self.head];
        self.head = (self.head + 1) % self.buf.len();
        self.len -= 1;
        if self.len == 0 {
            self.head = 0;
        }
        Some(v)
    }

    #[inline]
    pub(crate) fn push_back(&mut self, value: T) {
        if self.buf.is_empty() {
            self.buf = vec![T::default(); 8];
            self.head = 0;
            self.len = 0;
        } else if self.len == self.buf.len() {
            self.grow();
        }
        let idx = (self.head + self.len) % self.buf.len();
        self.buf[idx] = value;
        self.len += 1;
    }

    #[inline]
    pub(crate) fn iter(&self) -> RingIter<'_, T> {
        RingIter {
            buf: &self.buf,
            head: self.head,
            len: self.len,
            pos: 0,
        }
    }

    pub(crate) fn make_contiguous(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut self.buf[0..0];
        }
        if self.head + self.len <= self.buf.len() {
            let start = self.head;
            let end = start + self.len;
            return &mut self.buf[start..end];
        }
        let cap = self.buf.len().max(1);
        let mut new_buf = vec![T::default(); cap];
        for i in 0..self.len {
            let idx = (self.head + i) % self.buf.len();
            new_buf[i] = self.buf[idx];
        }
        self.buf = new_buf;
        self.head = 0;
        &mut self.buf[..self.len]
    }

    pub(crate) fn insert(&mut self, idx: usize, value: T) {
        if idx >= self.len {
            self.push_back(value);
            return;
        }
        if self.buf.is_empty() {
            self.buf = vec![T::default(); 8];
            self.head = 0;
            self.len = 0;
        }
        if self.len == self.buf.len() {
            self.grow();
        }
        self.make_contiguous();
        let len = self.len;
        self.buf.copy_within(idx..len, idx + 1);
        self.buf[idx] = value;
        self.len += 1;
    }

    fn grow(&mut self) {
        let old_cap = self.buf.len();
        let new_cap = if old_cap == 0 { 8 } else { old_cap * 2 };
        let mut new_buf = vec![T::default(); new_cap];
        for i in 0..self.len {
            let idx = (self.head + i) % old_cap;
            new_buf[i] = self.buf[idx];
        }
        self.buf = new_buf;
        self.head = 0;
    }
}

pub(crate) struct RingIter<'a, T: Copy + Default> {
    buf: &'a [T],
    head: usize,
    len: usize,
    pos: usize,
}

impl<'a, T: Copy + Default> Iterator for RingIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len || self.buf.is_empty() {
            return None;
        }
        let idx = (self.head + self.pos) % self.buf.len();
        self.pos += 1;
        Some(&self.buf[idx])
    }
}
