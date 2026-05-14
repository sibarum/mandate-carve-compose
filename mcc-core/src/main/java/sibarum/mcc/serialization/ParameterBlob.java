package sibarum.mcc.serialization;

import sibarum.mcc.serialization.GraphSchema.TensorEntry;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;

/**
 * Framed binary layout for the parameter blob (component.mcc/params.bin).
 *
 * <pre>
 *   header (16 bytes):
 *     magic       4 bytes  "MCC\\x01"
 *     version     int32    blob format version (= 1)
 *     tensorCount int32    number of tensors in this blob
 *     reserved    int32    set to 0
 *   body:
 *     contiguous little-endian f64 arrays, one tensor after another;
 *     each tensor's offset and length are recorded in the JSON
 *     {@link sibarum.mcc.serialization.GraphSchema.ParameterManifest}.
 * </pre>
 *
 * <p>SHA-256 of the body (excluding header) is stored in the manifest
 * for integrity verification at load time.
 */
public final class ParameterBlob {

    public static final byte[] MAGIC = { 'M', 'C', 'C', (byte) 0x01 };
    public static final int VERSION = 1;
    public static final int HEADER_BYTES = 16;
    public static final int BYTES_PER_DOUBLE = 8;

    private ParameterBlob() {}

    public record WriteResult(List<TensorEntry> entries, String sha256) {}

    /**
     * Write {@code tensors} to {@code out} in the framed format. Tensor
     * {@code offset} values are body-relative (after the header). The
     * returned SHA-256 covers the body only.
     */
    public static WriteResult write(List<NamedNodeTensor> tensors, OutputStream out) throws IOException {
        // Header.
        ByteBuffer header = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        header.put(MAGIC);
        header.putInt(VERSION);
        header.putInt(tensors.size());
        header.putInt(0); // reserved
        out.write(header.array());

        MessageDigest sha = sha256();
        long bodyOffset = 0L;
        List<TensorEntry> entries = new ArrayList<>(tensors.size());
        for (NamedNodeTensor t : tensors) {
            long bytes = (long) t.data().length * BYTES_PER_DOUBLE;
            ByteBuffer buf = ByteBuffer.allocate((int) bytes).order(ByteOrder.LITTLE_ENDIAN);
            for (double d : t.data()) buf.putDouble(d);
            byte[] arr = buf.array();
            out.write(arr);
            sha.update(arr);
            entries.add(new TensorEntry(t.node(), t.name(), t.shape(), bodyOffset, bytes));
            bodyOffset += bytes;
        }
        return new WriteResult(entries, hex(sha.digest()));
    }

    /**
     * Read the framed blob from {@code in} into a flat byte array of
     * the body (excluding the 16-byte header). Verifies the magic and
     * version; on success returns the body bytes and the recomputed
     * SHA-256 hex string.
     */
    public record ReadResult(byte[] body, String sha256, int tensorCount) {}

    public static ReadResult read(InputStream in) throws IOException {
        byte[] hdr = in.readNBytes(HEADER_BYTES);
        if (hdr.length != HEADER_BYTES) throw new IOException("truncated header");
        ByteBuffer h = ByteBuffer.wrap(hdr).order(ByteOrder.LITTLE_ENDIAN);
        byte[] mg = new byte[4];
        h.get(mg);
        for (int i = 0; i < 4; i++) {
            if (mg[i] != MAGIC[i]) throw new IOException("bad magic header in params blob");
        }
        int ver = h.getInt();
        if (ver != VERSION) throw new IOException("unsupported params blob version: " + ver);
        int count = h.getInt();
        h.getInt(); // reserved

        byte[] body = in.readAllBytes();
        MessageDigest sha = sha256();
        sha.update(body);
        return new ReadResult(body, hex(sha.digest()), count);
    }

    /** Decode a slice of the body bytes into a {@code double[]}. */
    public static double[] decode(byte[] body, long offset, long length) {
        if ((length % BYTES_PER_DOUBLE) != 0) {
            throw new IllegalArgumentException("tensor length not multiple of 8: " + length);
        }
        int n = (int) (length / BYTES_PER_DOUBLE);
        ByteBuffer buf = ByteBuffer.wrap(body, (int) offset, (int) length).order(ByteOrder.LITTLE_ENDIAN);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) out[i] = buf.getDouble();
        return out;
    }

    /** Pair carrying both the owning node id and the local parameter name. */
    public record NamedNodeTensor(String node, String name, int[] shape, double[] data) {}

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 not available", e);
        }
    }

    private static String hex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }
}
