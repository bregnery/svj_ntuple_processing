import os, os.path as osp, shutil, logging, math, uuid, tempfile, glob
import seutils # type: ignore


def setup_logger(name='hadd'):
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info('Logger %s is already defined', name)
    else:
        fmt = logging.Formatter(
            fmt = (
                '\033[34m[%(name)s:%(levelname)s:%(asctime)s:%(module)s:%(lineno)s]\033[0m'
                + ' %(message)s'
                ),
            datefmt='%Y-%m-%d %H:%M:%S'
            )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger
logger = setup_logger()


WORKDIR = 'WORKDIR'


def debug(flag=True):
    logger.setLevel(logging.DEBUG if flag else logging.INFO)


def uid():
    return str(uuid.uuid4())


def expand_wildcards(pats):
    expanded = []
    for pat in pats:
        if '*' in pat:
            if seutils.path.has_protocol(pat):
                expanded.extend(seutils.ls_wildcard(pat))
            else:
                expanded.extend(glob.glob(pat))
        else:
            expanded.append(pat)
    return expanded
                

def simple_hadd(root_files, dst, dry=False):
    """
    Basic hadd command: Formats and runs the hadd command for a list of
    input rootfiles and a destination.
    """
    if len(root_files) == 1:
        # No need for hadding, just copy
        rootfile = root_files[0]
        logger.debug('Just 1 rootfile - copying %s --> %s', rootfile, dst)
        if seutils.path.has_protocol(dst):
            seutils.cp(rootfile, dst)
        else:
            shutil.copyfile(rootfile, dst)
        return

    # Allow remote destinations, but only do the transfer at the end
    must_copy = False
    if seutils.path.has_protocol(dst):
        final_dst = dst
        dst = uid()+'.root'
        must_copy = True

    cmd = 'hadd -f {} {}'.format(dst, ' '.join(root_files))
    logger.warning('hadd command: ' + cmd)
    if dry: return
    try:
        # debug(True)
        rcode = os.system(cmd)
        if rcode != 0: raise Exception('Non-zero return code from hadd call')
    finally:
        # debug(False)
        pass

    if must_copy:
        logger.info('Staging out %s -> %s', dst, final_dst)
        seutils.cp(dst, final_dst)
        os.remove(dst)


def simple_hadd_worker(tup):
    """
    Just unpacks an input tuple and calls simple_hadd.
    Needed to work with multiprocessing.
    """
    return simple_hadd(*tup)


def ntuplenpz_concatenate(tup):
    import svj_ntuple_processing as svj
    input_files, dst = tup
    cols = [svj.Columns.load(f, encoding='latin1') for f in input_files]
    concatenated = svj.concat_columns(cols)
    concatenated.save(dst)


class PooledMerger:
    def __init__(self, input_files, dst, n_threads=6, chunk_size=200, workdir=WORKDIR):
        self.input_files = input_files
        self.dst = dst
        self.n_threads = n_threads
        self.chunk_size = chunk_size
        self.workdir = workdir
        self.stage = 1

    def copy(self):
        return PooledMerger(
            self.input_files,
            self.dst,
            self.n_threads,
            self.chunk_size,
            self.workdir,
            )

    @property
    def n_chunks(self):
        return int(math.ceil(len(self.input_files) / float(self.chunk_size)))

    @property
    def chunks(self):
        if not hasattr(self, '_chunks'):
            self._chunks = []
            for i in range(self.n_chunks):
                chunk = self.input_files[i*self.chunk_size : (i+1)*self.chunk_size]
                self._chunks.append(chunk)
        return self._chunks

    def hadd(self, dry=False):
        """
        Treat input files as root files, and run hadd on it
        """
        if self.n_chunks == 1:
            # No need for multiprocessing or a temporary workdir
            simple_hadd(self.input_files, self.dst, dry=dry)
            return
        
        import multiprocessing as mp

        # compile fn arguments, and store intermediate dsts
        fn_args = []
        intermediate_merged_files = []
        for chunk in self.chunks:
            dst = osp.join(self.workdir, uid()+'.root')
            fn_args.append((chunk, dst))
            intermediate_merged_files.append(dst)

        logger.info('STAGE %s; %s chunks, %s chunk_size', self.stage, self.n_chunks, self.chunk_size)
        if dry:
            for chunk, dst in fn_args:
                logger.debug('hadding %s --> %s', ' '.join(chunk), dst)
            return
        else:
            # Run hadd on the chunks
            with tempfile.TemporaryDirectory(self.workdir, dir='.') as tmpdir:
                p = mp.Pool(min(self.n_threads, self.n_chunks))
                p.map(simple_hadd_worker, fn_args)
                p.close()
                p.join()

        # Merge all the intermediate outputs
        pool = self.copy()
        pool.input_files = intermediate_merged_files
        pool.stage = self.stage+1
        pool.hadd(dry)


    def mergenpz(self, dry=False):
        if self.n_chunks == 1:
            # No need for multiprocessing or a temporary workdir
            if dry:
                logger.debug('merging %s --> %s', ' '.join(self.input_files), dst)
            else:
                ntuplenpz_concatenate([self.input_files, self.dst])
            return
        
        import multiprocessing as mp

        # compile fn arguments, and store intermediate dsts
        fn_args = []
        intermediate_merged_files = []
        for chunk in self.chunks:
            dst = osp.join(self.workdir, uid()+'.npz')
            fn_args.append((chunk, dst))
            intermediate_merged_files.append(dst)

        logger.info(
            'STAGE %s; %s chunks, %s chunk_size',
            self.stage, self.n_chunks, self.chunk_size
            )
        if dry:
            for chunk, dst in fn_args:
                logger.debug('merging %s --> %s', ' '.join(chunk), dst)
            return
        else:
            # Run hadd on the chunks
            with tempfile.TemporaryDirectory(self.workdir, dir='.') as tmpdir:
                p = mp.Pool(min(self.n_threads, self.n_chunks))
                p.map(ntuplenpz_concatenate, fn_args)
                p.close()
                p.join()

        # Merge all the intermediate outputs
        pool = self.copy()
        pool.input_files = intermediate_merged_files
        pool.stage = self.stage + 1
        pool.mergenpz(dry=dry)




def hadd_chunks(root_files, dst, n_threads=6, chunk_size=200, workdir='/tmp/tmphadd', dry=False):
    """
    Like hadd, but hadds a chunk of root files in threads to temporary files first,
    then hadds the temporary files into the final root file.
    The algorithm is recursive; if there are too many temporary files still, another intermediate
    chunked hadd is performed.
    """
    if not len(root_files): raise Exception('Need >0 rootfiles')
    if len(root_files) < chunk_size:
        # No need for chunking. This should also be the final step of the recursion
        simple_hadd(root_files, dst, dry=dry)
        return
    import multiprocessing as mp
    n_chunks = int(math.ceil(len(root_files) / float(chunk_size)))
    # Make a unique directory for temporary files
    workdir = osp.join(workdir, uid())
    os.makedirs(workdir)
    try:
        # debug(True)
        chunk_rootfiles = []
        # First compile list of function arguments
        func_args = []
        for i_chunk in range(n_chunks):
            chunk = root_files[ i_chunk*chunk_size : (i_chunk+1)*chunk_size ]
            chunk_dst = osp.join(workdir, uid()+'.root')
            func_args.append([chunk, chunk_dst, dry])
            chunk_rootfiles.append(chunk_dst)
            if dry: logger.debug('hadding %s --> %s', ' '.join(chunk), chunk_dst)
        # Submit to multiprocessing in one go:
        if not dry:
            n_threads = min(n_threads, n_chunks) # Don't start more threads if not necessary
            p = mp.Pool(n_threads)
            p.map(simple_hadd_worker, func_args)
            p.close()
            p.join()
        # Merge the chunks into the final destination, potentially with another chunked merge
        hadd_chunks(chunk_rootfiles, dst, n_threads, chunk_size, workdir, dry)
    finally:
        logger.warning('Removing %s', workdir)
        shutil.rmtree(workdir)
        # debug(False)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dst', nargs=1, type=str, required=True)
    parser.add_argument('rootfiles', nargs='+', type=str)
    parser.add_argument('-n', '--nthreads', default=10, type=int)
    parser.add_argument('-c', '--chunksize', default=10, type=int)
    args = parser.parse_args()

    rootfiles = expand_wildcards(args.rootfiles)

    with tempfile.TemporaryDirectory('hadd', dir='.') as tmpdir:

        # Allow remote destinations, but only do the transfer at the end
        must_copy = False
        dst = args.dst[0]
        if seutils.path.has_protocol(dst):
            final_dst = dst
            dst = osp.join(tmpdir, 'out.root')
            must_copy = True

        hadd_chunks(rootfiles, dst, args.nthreads, args.chunksize, tmpdir)

        if must_copy:
            logger.info('Staging out %s -> %s', dst, final_dst)
            seutils.cp(dst, final_dst)
